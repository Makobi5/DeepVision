import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
import os
import numpy as np
import cv2
from PIL import Image
from sklearn.utils import class_weight
from collections import Counter
import torch.nn.functional as F

class VideoDataset(Dataset):
    def __init__(self, video_data, transform=None, frame_count=10, split='train', temporal_features=True, 
                 augment_minority=False):
        """
        Args:
            video_data (dict): Dictionary of video paths returned by prepare_classification_dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            frame_count (int): Number of frames to extract per video.
            split (str): 'train', 'test', or 'val'.
            temporal_features (bool): Whether to return temporal features (sequence of frames) or averaged frames.
            augment_minority (bool): Whether to apply extra augmentation to minority classes.
        """
        self.video_paths = []
        self.labels = []
        self.transform = transform
        self.frame_count = frame_count
        self.temporal_features = temporal_features
        self.augment_minority = augment_minority

        # Check if the split is valid
        if split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'")

        self.split = split
        self.class_names = list(video_data[split].keys())  # Extract class names

        # Populate video_paths and labels from the video_data dictionary
        for class_name in self.class_names:
            for video_path in video_data[split][class_name]:
                self.video_paths.append(video_path)
                self.labels.append(class_name)

        # Create class_to_idx mapping
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_names)}
        
        # Print class distribution
        label_counts = Counter(self.labels)
        print(f"Class distribution in {split} set:")
        for class_name, count in label_counts.items():
            print(f"  {class_name}: {count}")
        
        # Store label counts for augmentation decisions
        self.label_counts = label_counts

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        label_idx = self.class_to_idx[label]
        
        # Determine if this sample belongs to a minority class
        # We'll consider Violence and Weaponized as minority classes based on metrics
        is_minority_class = label in ['Violence', 'Weaponized']
        
        frames = self._extract_frames(video_path, self.frame_count)

        if len(frames) > 0:
            # If we have frames, process them
            processed_frames = []
            for frame in frames:
                # Convert to RGB (OpenCV loads as BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                # Apply stronger augmentation for minority classes in training mode
                if self.transform:
                    if self.split == 'train' and self.augment_minority and is_minority_class:
                        # Apply additional augmentation for minority classes
                        frame = self._apply_extra_augmentation(frame)
                    frame = self.transform(frame)
                    
                processed_frames.append(frame)
            
            # Stack frames along a new dimension
            if len(processed_frames) > 0:
                stacked_frames = torch.stack(processed_frames)
                
                if self.temporal_features:
                    # Return sequence of frames for temporal processing
                    # If we don't have enough frames, repeat the last frame
                    if stacked_frames.size(0) < self.frame_count:
                        last_frame = stacked_frames[-1].unsqueeze(0)
                        padding = last_frame.repeat(self.frame_count - stacked_frames.size(0), 1, 1, 1)
                        stacked_frames = torch.cat([stacked_frames, padding], dim=0)
                    # If we have too many frames, take evenly spaced frames
                    elif stacked_frames.size(0) > self.frame_count:
                        indices = np.linspace(0, stacked_frames.size(0) - 1, self.frame_count, dtype=int)
                        stacked_frames = stacked_frames[indices]
                    
                    return stacked_frames, label_idx
                else:
                    # Average the features across frames for non-temporal processing
                    avg_frame = torch.mean(stacked_frames, dim=0)
                    return avg_frame, label_idx
            else:
                print(f"Warning: No processed frames from {video_path}")
                if self.temporal_features:
                    # Return zero tensor of shape [frame_count, channels, height, width]
                    return torch.zeros((self.frame_count, 3, 224, 224)), label_idx
                else:
                    return torch.zeros((3, 224, 224)), label_idx
        else:
            print(f"Warning: No frames extracted from {video_path}")
            if self.temporal_features:
                # Return zero tensor of shape [frame_count, channels, height, width]
                return torch.zeros((self.frame_count, 3, 224, 224)), label_idx
            else:
                return torch.zeros((3, 224, 224)), label_idx

    def _apply_extra_augmentation(self, image):
        """Apply extra augmentation for minority classes"""
        from PIL import ImageEnhance, ImageFilter
        import random
        
        # Random brightness and contrast variation (more extreme)
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Brightness(image).enhance(factor)
        
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Contrast(image).enhance(factor)
            
        # Random color variation
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Color(image).enhance(factor)
        
        # Random sharpness variation
        if random.random() > 0.5:
            factor = random.uniform(0.0, 2.0)
            image = ImageEnhance.Sharpness(image).enhance(factor)
        
        # Random application of filters
        if random.random() > 0.7:
            filter_choice = random.choice([
                ImageFilter.BLUR,
                ImageFilter.DETAIL,
                ImageFilter.EDGE_ENHANCE,
                ImageFilter.SMOOTH
            ])
            image = image.filter(filter_choice)
            
        return image

    def _extract_frames(self, video_path, frame_count):
        """
        Extracts a specified number of evenly spaced frames from a video.
        Uses temporal information by capturing motion between frames.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"Error: Could not read frames from {video_path}")
            cap.release()
            return frames
        
        # Calculate frame indices for even spacing
        if frame_count > total_frames:
            frame_indices = list(range(total_frames))
        else:
            # Take evenly spaced frames but ensure we capture the beginning, middle, and end
            # This helps with detecting action/motion patterns specific to each class
            frame_indices = np.linspace(0, total_frames-1, frame_count, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                
        cap.release()
        return frames


class TemporalCNN(nn.Module):
    """An improved CNN model that processes temporal information from video frames."""
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(TemporalCNN, self).__init__()
        # Use EfficientNet B2 for better feature extraction
        self.base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        # Remove the classifier
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Feature dimension from efficientnet_b2
        feature_dim = 1408
        
        # Improved temporal feature processing with more layers
        self.temporal_conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=768, kernel_size=3, padding=1)
        self.temporal_bn1 = nn.BatchNorm1d(768)
        self.temporal_conv2 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.temporal_bn2 = nn.BatchNorm1d(512)
        
        # Use both max and average pooling and concat results
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Attention mechanism for temporal features
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Final classifier with more gradual reduction
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, num_frames, channels, height, width]
        batch_size, num_frames, c, h, w = x.size()
        
        # Reshape to process each frame
        x = x.view(batch_size * num_frames, c, h, w)
        
        # Extract features for each frame
        x = self.features(x)
        
        # Reshape back to [batch_size, num_frames, features]
        x = x.view(batch_size, num_frames, -1)
        
        # Transpose to [batch_size, features, num_frames] for 1D convolution
        x = x.transpose(1, 2)
        
        # Apply improved temporal convolutions
        x = F.relu(self.temporal_bn1(self.temporal_conv1(x)))
        x_conv = F.relu(self.temporal_bn2(self.temporal_conv2(x)))
        
        # Apply attention mechanism along temporal dimension
        # First, prepare for attention
        x_attn = x_conv.transpose(1, 2)  # [batch_size, num_frames, features]
        attn_weights = self.attention(x_attn).squeeze(-1)  # [batch_size, num_frames]
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(2)  # [batch_size, num_frames, 1]
        
        # Apply attention weights
        x_weighted = x_attn * attn_weights  # [batch_size, num_frames, features]
        x_weighted = x_weighted.transpose(1, 2)  # [batch_size, features, num_frames]
        
        # Apply both pooling methods and concatenate
        x_max = self.max_pool(x_weighted).squeeze(-1)  # [batch_size, features]
        x_avg = self.avg_pool(x_conv).squeeze(-1)  # [batch_size, features]
        
        # Concatenate the pooled features
        x_combined = torch.cat([x_max, x_avg], dim=1)  # [batch_size, features*2]
        
        # Classifier
        x = self.classifier(x_combined)
        
        return x


class FeatureEnhancedCNN(nn.Module):
    """A model specifically designed to distinguish Weaponized content from Normal content."""
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(FeatureEnhancedCNN, self).__init__()
        self.base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        # Remove the classifier
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Feature dimension from efficientnet_b2
        feature_dim = 1408
        
        # Create a spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Create a channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 16, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Edge detection layer (for weapon detection)
        self.edge_detect = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
        # Initialize with Sobel filters
        sobel_kernels = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],  # Horizontal
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],  # Vertical
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]  # Laplacian
        ], dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            self.edge_detect.weight.copy_(sobel_kernels.repeat(1, 1, 1, 1))
            self.edge_detect.weight.requires_grad = False
            
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Apply edge detection
        edge_features = self.edge_detect(x)
        
        # Process through base CNN
        features = self.features(x)
        
        # Apply attention mechanisms
        channel_weights = self.channel_attention(features)
        weighted_features = features * channel_weights
        
        spatial_weights = self.spatial_attention(features)
        weighted_features = weighted_features * spatial_weights
        
        # Global average pooling
        pooled_features = F.adaptive_avg_pool2d(weighted_features, 1).flatten(1)
        
        # Process edge features
        edge_processed = self.base_model.features(edge_features)
        edge_pooled = F.adaptive_avg_pool2d(edge_processed, 1).flatten(1)
        
        # Concatenate regular and edge features
        combined_features = torch.cat([pooled_features, edge_pooled], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output


class TwoStageModel:
    def __init__(self, model_path=None, video_data=None, dropout_rate=0.7, 
                 frame_count=10, use_weighted_sampler=True, class_weights_manual=None,
                 augment_minority=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.dropout_rate = dropout_rate
        self.frame_count = frame_count
        self.use_weighted_sampler = use_weighted_sampler
        self.augment_minority = augment_minority
        self.class_weights_manual = class_weights_manual
        
        # Define transforms with stronger augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-20, 20)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3)
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize datasets for both stages
        # For the first stage: Normal vs Not Normal (Violence + Weaponized)
        self._prepare_first_stage_data(video_data)
        
        # For the second stage: Violence vs Weaponized
        self._prepare_second_stage_data(video_data)
        
        # For the feature-enhanced model: Normal vs Weaponized
        self._prepare_weaponized_detector_data(video_data)
        
        # Initialize models
        self.first_stage_model = self._create_model(num_classes=2)  # Normal vs Not Normal
        self.second_stage_model = TemporalCNN(num_classes=2, dropout_rate=dropout_rate)  # Violence vs Weaponized
        self.weaponized_detector = FeatureEnhancedCNN(num_classes=2, dropout_rate=dropout_rate)  # Normal vs Weaponized
        
        # Move models to device
        self.first_stage_model = self.first_stage_model.to(self.device)
        self.second_stage_model = self.second_stage_model.to(self.device)
        self.weaponized_detector = self.weaponized_detector.to(self.device)
        
        # Load pre-trained weights if provided
        if model_path and isinstance(model_path, dict):
            if 'first_stage' in model_path and os.path.exists(model_path['first_stage']):
                self.first_stage_model.load_state_dict(torch.load(model_path['first_stage'], map_location=self.device))
                print(f"Loaded first stage weights from {model_path['first_stage']}")
            
            if 'second_stage' in model_path and os.path.exists(model_path['second_stage']):
                self.second_stage_model.load_state_dict(torch.load(model_path['second_stage'], map_location=self.device))
                print(f"Loaded second stage weights from {model_path['second_stage']}")
                
            if 'weaponized_detector' in model_path and os.path.exists(model_path['weaponized_detector']):
                self.weaponized_detector.load_state_dict(torch.load(model_path['weaponized_detector'], map_location=self.device))
                print(f"Loaded weaponized detector weights from {model_path['weaponized_detector']}")

    def _prepare_first_stage_data(self, video_data):
        """Prepare data for the first stage: Normal vs Not Normal"""
        # Create a modified version of video_data for binary classification
        binary_video_data = {'train': {}, 'test': {}}
        
        # Normal class stays as is
        binary_video_data['train']['Normal'] = video_data['train']['Normal']
        binary_video_data['test']['Normal'] = video_data['test']['Normal']
        
        # Combine Violence and Weaponized into "Not Normal"
        binary_video_data['train']['Not_Normal'] = []
        binary_video_data['test']['Not_Normal'] = []
        
        for split in ['train', 'test']:
            for class_name in ['Violence', 'Weaponized']:
                if class_name in video_data[split]:
                    binary_video_data[split]['Not_Normal'].extend(video_data[split][class_name])
        
        # Initialize datasets
        self.first_stage_train_dataset = VideoDataset(
            binary_video_data, transform=self.train_transform, 
            frame_count=self.frame_count, split='train', temporal_features=False,
            augment_minority=self.augment_minority
        )
        
        self.first_stage_test_dataset = VideoDataset(
            binary_video_data, transform=self.test_transform, 
            frame_count=self.frame_count, split='test', temporal_features=False
        )
        
        # Create weighted sampler for training data
        if self.use_weighted_sampler:
            class_counts = Counter([self.first_stage_train_dataset.class_to_idx[label] 
                                   for label in self.first_stage_train_dataset.labels])
            
            # Override with manual weights if provided
            if self.class_weights_manual and 'first_stage' in self.class_weights_manual:
                class_weights = self.class_weights_manual['first_stage']
            else:
                # Use adjusted weights to emphasize "Not Normal" class
                class_weights = {0: 0.8, 1: 1.2}  # Normal: 0.8, Not_Normal: 1.2
                
            print(f"Using first stage class weights: {class_weights}")
            sample_weights = [class_weights[self.first_stage_train_dataset.class_to_idx[label]] 
                             for label in self.first_stage_train_dataset.labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        # Initialize DataLoaders
        self.first_stage_train_loader = DataLoader(
            self.first_stage_train_dataset, batch_size=32, shuffle=shuffle,
            sampler=sampler, num_workers=4, pin_memory=True
        )
        
        self.first_stage_test_loader = DataLoader(
            self.first_stage_test_dataset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True
        )

    def _prepare_second_stage_data(self, video_data):
        """Prepare data for the second stage: Violence vs Weaponized"""
        # Create a filtered version of video_data for Violence vs Weaponized
        filtered_video_data = {'train': {}, 'test': {}}
        
        # Only include Violence and Weaponized classes
        for split in ['train', 'test']:
            for class_name in ['Violence', 'Weaponized']:
                if class_name in video_data[split]:
                    filtered_video_data[split][class_name] = video_data[split][class_name]
        
        # Initialize datasets with temporal features
        self.second_stage_train_dataset = VideoDataset(
            filtered_video_data, transform=self.train_transform, 
            frame_count=self.frame_count, split='train', temporal_features=True,
            augment_minority=self.augment_minority
        )
        
        self.second_stage_test_dataset = VideoDataset(
            filtered_video_data, transform=self.test_transform, 
            frame_count=self.frame_count, split='test', temporal_features=True
        )
        
        # Create weighted sampler for training data
        if self.use_weighted_sampler:
            class_counts = Counter([self.second_stage_train_dataset.class_to_idx[label] 
                                   for label in self.second_stage_train_dataset.labels])
            
            # Override with manual weights if provided
            if self.class_weights_manual and 'second_stage' in self.class_weights_manual:
                class_weights = self.class_weights_manual['second_stage']
            else:
                # Use balanced weights
                class_weights = {class_idx: 1.0 / count for class_idx, count in class_counts.items()}
                
            print(f"Using second stage class weights: {class_weights}")
            sample_weights = [class_weights[self.second_stage_train_dataset.class_to_idx[label]] 
                             for label in self.second_stage_train_dataset.labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        # Initialize DataLoaders with smaller batch size for more updates
        self.second_stage_train_loader = DataLoader(
            self.second_stage_train_dataset, batch_size=12, shuffle=shuffle,
            sampler=sampler, num_workers=4, pin_memory=True
        )
        
        self.second_stage_test_loader = DataLoader(
            self.second_stage_test_dataset, batch_size=12, shuffle=False,
            num_workers=4, pin_memory=True
        )

    def _prepare_weaponized_detector_data(self, video_data):
        """Prepare data for the weaponized detector: Normal vs Weaponized"""
        # Create a filtered version of video_data for Normal vs Weaponized
        filtered_video_data = {'train': {}, 'test': {}}
        
        # Only include Normal and Weaponized classes
        for split in ['train', 'test']:
            for class_name in ['Normal', 'Weaponized']:
                if class_name in video_data[split]:
                    filtered_video_data[split][class_name] = video_data[split][class_name]
        
        # Initialize datasets
        self.weaponized_train_dataset = VideoDataset(
            filtered_video_data, transform=self.train_transform, 
            frame_count=self.frame_count, split='train', temporal_features=False,
            augment_minority=self.augment_minority
        )
        
        self.weaponized_test_dataset = VideoDataset(
            filtered_video_data, transform=self.test_transform, 
            frame_count=self.frame_count, split='test', temporal_features=False
        )
        
        # Create weighted sampler for training data
        if self.use_weighted_sampler:
            class_counts = Counter([self.weaponized_train_dataset.class_to_idx[label] 
                                   for label in self.weaponized_train_dataset.labels])
            
            # Override with manual weights if provided
            if self.class_weights_manual and 'weaponized_detector' in self.class_weights_manual:
                class_weights = self.class_weights_manual['weaponized_detector']
            else:
                # Calculate balanced class weights
                total_samples = sum(class_counts.values())
                # Adjust weights to emphasize the Weaponized class
                weaponized_idx = self.weaponized_train_dataset.class_to_idx['Weaponized']
                normal_idx = self.weaponized_train_dataset.class_to_idx['Normal']
                class_weights = {
                    normal_idx: 0.7,
                    weaponized_idx: 1.3
                }
                
            print(f"Using weaponized detector class weights: {class_weights}")
            sample_weights = [class_weights[self.weaponized_train_dataset.class_to_idx[label]] 
                             for label in self.weaponized_train_dataset.labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        # Initialize DataLoaders
        self.weaponized_train_loader = DataLoader(
            self.weaponized_train_dataset, batch_size=24, shuffle=shuffle,
            sampler=sampler, num_workers=4, pin_memory=True
        )
        
        self.weaponized_test_loader = DataLoader(
            self.weaponized_test_dataset, batch_size=24, shuffle=False,
            num_workers=4, pin_memory=True
        )

    def _create_model(self, num_classes):
        """Create a model for the first stage with improved architecture"""
        # Use EfficientNet B2 for better feature extraction
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        
        # Create a more robust classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, num_classes)
        )
        return model

    def train_first_stage(self, epochs=30, learning_rate=0.0003, weight_decay=3e-4, patience=12, 
                         label_smoothing=0.1):
        """Train the first stage model (Normal vs Not Normal)"""
        print("Training first stage model: Normal vs Not Normal")
        
        # Compute or use provided class weights for loss function
        if self.class_weights_manual and 'first_stage_loss' in self.class_weights_manual:
    class_weights = torch.tensor(
        self.class_weights_manual['first_stage_loss'], dtype=torch.float
    ).to(self.device)
else:  # <- This was missing
    # Compute class weights from dataset
    y_train = np.array([self.first_stage_train_dataset.class_to_idx[label] 
                       for label in self.first_stage_train_dataset.labels])
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
    
    # Adjust weights to favor the Not Normal class (as per recommendation)
    class_weights[1] *= 1.2