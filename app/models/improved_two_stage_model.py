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
        self.first_stage_frame_count = 8

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
                 frame_count=10, first_stage_frame_count=8, use_weighted_sampler=True, class_weights_manual=None,
                 augment_minority=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.dropout_rate = dropout_rate
        self.frame_count = frame_count
        self.first_stage_frame_count = first_stage_frame_count # Initialize first_stage_frame_count
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
        if video_data is not None:
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
            frame_count=self.first_stage_frame_count, split='train', temporal_features=False,
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
                # Use more balanced weights for first stage
                class_weights = {0: 0.9, 1: 1.1}  # Normal: 0.9, Not_Normal: 1.1  # Normal: 0.8, Not_Normal: 1.2

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
            if self.class_weights_manual and 'second_stage' in self.class_weights_manual and self.class_weights_manual['second_stage'] is not None:
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
        """Create a model for the first stage with original architecture"""
        # Use EfficientNet B0 for first stage as in original model (not B2)
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features

        # Revert to simpler classifier with original dropout of 0.6
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.6),  # Use 0.6 instead of self.dropout_rate
            nn.Linear(num_ftrs, num_classes)
        )
        return model

    def train_first_stage(self, epochs=30, learning_rate=0.0003, weight_decay=3e-4, patience=12,
                         label_smoothing=0.1):
        first_stage_dropout = 0.6
        first_stage_weight_decay = 2e-4
        """Train the first stage model (Normal vs Not Normal)"""
        print("Training first stage model: Normal vs Not Normal")

        # Compute or use provided class weights for loss function
        # Compute or use provided class weights for loss function
        if self.class_weights_manual and 'first_stage_loss' in self.class_weights_manual:
            # Use manually provided weights
            class_weights = torch.tensor(
                self.class_weights_manual['first_stage_loss'], dtype=torch.float
            ).to(self.device)
        else:
            # Compute class weights from dataset
            y_train = np.array([self.first_stage_train_dataset.class_to_idx[label]
                            for label in self.first_stage_train_dataset.labels])
            computed_weights = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train
            )
            class_weights = torch.tensor(computed_weights, dtype=torch.float).to(self.device)

        print(f"First stage class weights: {class_weights}")

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(self.first_stage_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Improved learning rate scheduler with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=learning_rate/20
        )

        # Initialize tracking variables
        best_val_acc = 0.0
        no_improve_counter = 0
        best_model_path = os.path.join('models', 'first_stage_best.pth')

        # Make sure models directory exists
        os.makedirs('models', exist_ok=True)

        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []
        }

        # Training loop
        for epoch in range(epochs):
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # Training phase
            self.first_stage_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, labels) in enumerate(self.first_stage_train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.first_stage_model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.first_stage_model.parameters(), max_norm=1.0)
                optimizer.step()

                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"First Stage - Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(self.first_stage_train_loader)} | "
                          f"Loss: {loss.item():.4f}")

            # Calculate average training loss and accuracy
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation phase
            self.first_stage_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in self.first_stage_test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Forward pass
                    outputs = self.first_stage_model(inputs)
                    loss = criterion(outputs, labels)

                    # Track statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate average validation loss and accuracy
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # Update scheduler
            scheduler.step()

            # Check if this is the best model
            if val_acc > best_val_acc:
                print(f"First Stage - Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                no_improve_counter = 0
                torch.save(self.first_stage_model.state_dict(), best_model_path)
                print(f"Saved best first stage model at epoch {epoch+1}")
            else:
                no_improve_counter += 1

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print statistics
            print(f"First Stage - Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")

            # Early stopping check
            if no_improve_counter >= patience:
                print(f"First Stage - Early stopping triggered at epoch {epoch+1}")
                break

        # Load the best model
        print(f"Loading best first stage model")
        self.first_stage_model.load_state_dict(torch.load(best_model_path))
        self.first_stage_model.eval()

        return history, best_model_path

    def train_second_stage(self, epochs=50, learning_rate=0.0002, weight_decay=3e-4, patience=15,
                          label_smoothing=0.1):
        """Train the second stage model (Violence vs Weaponized)"""
        print("Training second stage model: Violence vs Weaponized")

        # Compute or use provided class weights for loss function
        if self.class_weights_manual and 'second_stage_loss' in self.class_weights_manual and self.class_weights_manual['second_stage_loss'] is not None:
            class_weights = torch.tensor(
                self.class_weights_manual['second_stage_loss'], dtype=torch.float
            ).to(self.device)
        else:
            # Compute class weights from dataset
            y_train = np.array([self.second_stage_train_dataset.class_to_idx[label]
                               for label in self.second_stage_train_dataset.labels])
            class_weights = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

            # Further adjust weights to improve discrimination between Violence and Weaponized
            violence_idx = self.second_stage_train_dataset.class_to_idx.get('Violence', 0)
            class_weights[violence_idx] *= 1.2  # Increase weight for Violence class

        print(f"Second stage class weights: {class_weights}")

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(self.second_stage_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=learning_rate/20
        )

        # Initialize tracking variables
        best_val_acc = 0.0
        no_improve_counter = 0
        best_model_path = os.path.join('models', 'second_stage_best.pth')

        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []
        }

        # Training loop
        for epoch in range(epochs):
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # Training phase
            self.second_stage_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, labels) in enumerate(self.second_stage_train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.second_stage_model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.second_stage_model.parameters(), max_norm=1.0)
                optimizer.step()

                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # Print progress every 5 batches (since batch size is smaller)
                if (batch_idx + 1) % 5 == 0:
                    print(f"Second Stage - Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(self.second_stage_train_loader)} | "
                          f"Loss: {loss.item():.4f}")

            # Calculate average training loss and accuracy
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation phase
            self.second_stage_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in self.second_stage_test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Forward pass
                    outputs = self.second_stage_model(inputs)
                    loss = criterion(outputs, labels)

                    # Track statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate average validation loss and accuracy
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # Update scheduler
            scheduler.step()

            # Check if this is the best model
            if val_acc > best_val_acc:
                print(f"Second Stage - Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                no_improve_counter = 0
                torch.save(self.second_stage_model.state_dict(), best_model_path)
                print(f"Saved best second stage model at epoch {epoch+1}")
            else:
                no_improve_counter += 1

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print statistics
            print(f"Second Stage - Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")

            # Early stopping check
            if no_improve_counter >= patience:
                print(f"Second Stage - Early stopping triggered at epoch {epoch+1}")
                break

        # Load the best model
            print(f"Loading best second stage model")
            self.second_stage_model.load_state_dict(torch.load(best_model_path))
            self.second_stage_model.eval()

            return history, best_model_path

    def train_weaponized_detector(self, epochs=40, learning_rate=0.0002, weight_decay=3e-4, patience=15,
                              label_smoothing=0.1):
        """Train the dedicated weaponized detector model (Normal vs Weaponized)"""
        print("Training weaponized detector model: Normal vs Weaponized")

        # Compute or use provided class weights for loss function
        if self.class_weights_manual and 'weaponized_detector_loss' in self.class_weights_manual and self.class_weights_manual['weaponized_detector_loss'] is not None:
            class_weights = torch.tensor(
                self.class_weights_manual['weaponized_detector_loss'], dtype=torch.float
            ).to(self.device)
        else:
            # Compute class weights from dataset
            y_train = np.array([self.weaponized_train_dataset.class_to_idx[label]
                            for label in self.weaponized_train_dataset.labels])
            class_weights = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

            # Adjust weights to favor detecting Weaponized content
            weaponized_idx = self.weaponized_train_dataset.class_to_idx.get('Weaponized', 1)
            class_weights[weaponized_idx] *= 1.4  # Significantly increase weight for Weaponized class

        print(f"Weaponized detector class weights: {class_weights}")

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(self.weaponized_detector.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=8, T_mult=2, eta_min=learning_rate/20
        )

        # Initialize tracking variables
        best_val_acc = 0.0
        no_improve_counter = 0
        best_model_path = os.path.join('models', 'weaponized_detector_best.pth')

        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []
        }

        # Training loop
        for epoch in range(epochs):
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            # Training phase
            self.weaponized_detector.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, labels) in enumerate(self.weaponized_train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.weaponized_detector(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.weaponized_detector.parameters(), max_norm=1.0)
                optimizer.step()

                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # Print progress every 5 batches
                if (batch_idx + 1) % 5 == 0:
                    print(f"Weaponized Detector - Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(self.weaponized_train_loader)} | "
                        f"Loss: {loss.item():.4f}")

            # Calculate average training loss and accuracy
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation phase
            self.weaponized_detector.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in self.weaponized_test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Forward pass
                    outputs = self.weaponized_detector(inputs)
                    loss = criterion(outputs, labels)

                    # Track statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate average validation loss and accuracy
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # Update scheduler
            scheduler.step()

            # Check if this is the best model
            if val_acc > best_val_acc:
                print(f"Weaponized Detector - Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                no_improve_counter = 0
                torch.save(self.weaponized_detector.state_dict(), best_model_path)
                print(f"Saved best weaponized detector model at epoch {epoch+1}")
            else:
                no_improve_counter += 1

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print statistics
            print(f"Weaponized Detector - Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.6f}")

            # Early stopping check
            if no_improve_counter >= patience:
                print(f"Weaponized Detector - Early stopping triggered at epoch {epoch+1}")
                break

        # Load the best model
        print(f"Loading best weaponized detector model")
        self.weaponized_detector.load_state_dict(torch.load(best_model_path))
        self.weaponized_detector.eval()

        return history, best_model_path

    def train(self, epochs_first=30, epochs_second=50, epochs_weaponized=40,
          learning_rate_first=0.0003, learning_rate_second=0.0002, learning_rate_weaponized=0.0002,
          weight_decay_first=2e-4, weight_decay_second=3e-4, weight_decay_weaponized=3e-4,
          patience_first=10, patience_second=15, patience_weaponized=15,
          label_smoothing=0.1):
        """Train all models in the ensemble approach"""

        first_stage_history, first_stage_path = self.train_first_stage(
            epochs=epochs_first, learning_rate=learning_rate_first,
            weight_decay=weight_decay_first, patience=patience_first,
            label_smoothing=label_smoothing
    )

        second_stage_history, second_stage_path = self.train_second_stage(
            epochs=epochs_second, learning_rate=learning_rate_second,
            weight_decay=weight_decay_second, patience=patience_second,
            label_smoothing=label_smoothing
    )

    # Train weaponized detector
        weaponized_history, weaponized_path = self.train_weaponized_detector(
            epochs=epochs_weaponized, learning_rate=learning_rate_weaponized,
            weight_decay=weight_decay_weaponized, patience=patience_weaponized,
            label_smoothing=label_smoothing
    )

    # Return combined history and paths
        combined_history = {
            'first_stage': first_stage_history,
            'second_stage': second_stage_history,
            'weaponized_detector': weaponized_history
    }

        model_paths = {
            'first_stage': first_stage_path,
            'second_stage': second_stage_path,
            'weaponized_detector': weaponized_path
    }

        return combined_history, model_paths

    def evaluate(self, verbose=True, use_ensemble=True):
        """Evaluate the improved ensemble model on the test set"""
        self.first_stage_model.eval()
        self.second_stage_model.eval()
        self.weaponized_detector.eval()

        # Original class names and mapping for final predictions
        original_classes = ['Normal', 'Violence', 'Weaponized']
        class_mapping = {
            (0, None): 0,  # Normal
            (1, 0): 1,     # Not Normal -> Violence
            (1, 1): 2      # Not Normal -> Weaponized
        }

        # Get all test data from the original datasets
        all_video_paths = []
        all_true_labels = []

        # Collect paths and true labels from first stage test dataset
        for i in range(len(self.first_stage_test_dataset)):
            path = self.first_stage_test_dataset.video_paths[i]
            label = self.first_stage_test_dataset.labels[i]

            # Map binary labels back to original classes
            if label == 'Normal':
                true_label = 0  # Normal
            else:  # Not_Normal
                # Find the original label in the source data
                if 'Violence' in path.lower() or (path in self.second_stage_test_dataset.video_paths and
                   self.second_stage_test_dataset.labels[self.second_stage_test_dataset.video_paths.index(path)] == 'Violence'):
                    true_label = 1  # Violence
                else:
                    true_label = 2  # Weaponized

            all_video_paths.append(path)
            all_true_labels.append(true_label)

        # Create confusion matrix
        confusion_matrix = torch.zeros(3, 3, device=self.device)

        # Track all predictions
        all_predictions = []

        # Process each video
        for i, path in enumerate(all_video_paths):
            true_label = all_true_labels[i]

            # Load and preprocess the frames
            frames = self._extract_frames_for_prediction(path)

            if use_ensemble:
                # Ensemble approach
                # 1. Get predictions from all models
                first_stage_pred = self._predict_first_stage(frames)
                second_stage_pred = self._predict_second_stage(frames)
                weaponized_pred = self._predict_weaponized(frames)

                # 2. If first stage predicts Normal, we trust it unless weaponized_detector strongly disagrees
                if first_stage_pred == 0:
                    weaponized_confidence = self._get_weaponized_confidence(frames)

                    # If weaponized detector is very confident (>0.85), override to Weaponized
                    if weaponized_confidence > 0.85:
                        final_pred = 2  # Weaponized
                    else:
                        final_pred = 0  # Normal
                else:
                    # If Not Normal, use both second stage and weaponized detector
                    # to distinguish between Violence and Weaponized
                    second_stage_confidence = self._get_second_stage_confidence(frames, second_stage_pred)
                    weaponized_confidence = self._get_weaponized_confidence(frames)

                    # If weaponized detector has high confidence, trust it
                    if weaponized_confidence > 0.8:
                        final_pred = 2  # Weaponized
                    elif second_stage_pred == 0:
                        final_pred = 1  # Violence
                    else:
                        final_pred = 2  # Weaponized
            else:
                # Original two-stage approach
                # First stage prediction (Normal vs Not Normal)
                first_stage_pred = self._predict_first_stage(frames)

                # If Normal, we're done
                if first_stage_pred == 0:
                    final_pred = 0
                else:
                    # If Not Normal, use second stage to distinguish between Violence and Weaponized
                    second_stage_pred = self._predict_second_stage(frames)

                    # Map to final prediction
                    final_pred = class_mapping[(1, second_stage_pred)]

            # Update confusion matrix
            confusion_matrix[true_label, final_pred] += 1
            all_predictions.append(final_pred)

        # Calculate accuracy
        correct = (confusion_matrix.diag().sum() / confusion_matrix.sum()).item()

        if verbose:
            print(f"Overall Accuracy: {correct:.4f}")

            # Calculate per-class metrics
            print("\nPer-Class Metrics:")

            for i, class_name in enumerate(original_classes):
                # Calculate true positives, false positives, and false negatives
                true_pos = confusion_matrix[i, i].item()
                false_pos = confusion_matrix[:, i].sum().item() - true_pos
                false_neg = confusion_matrix[i, :].sum().item() - true_pos

                # Calculate precision, recall, and F1 score
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                print(f"{class_name}:")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall:    {recall:.4f}")
                print(f"  F1-Score:  {f1:.4f}")

            # Print confusion matrix
            print("\nConfusion Matrix:")
            print("Predicted →")
            print("            " + " ".join(f"{name:>10}" for name in original_classes))
            print("Actual ↓")

            for i, name in enumerate(original_classes):
                row = f"{name:10}" + " ".join([f"{int(confusion_matrix[i, j].item()):10d}" for j in range(3)])
                print(row)

        # Return evaluation metrics
        return {
            'accuracy': correct,
            'confusion_matrix': confusion_matrix.cpu().numpy(),
            'predictions': all_predictions,
            'true_labels': all_true_labels
    }
    def classify(self, video_path, use_ensemble=True):
        """Classify a single video using the ensemble approach"""
        original_classes = ['Normal', 'Violence', 'Weaponized']
        class_mapping = {
            (0, None): 0,  # Normal
            (1, 0): 1,     # Not Normal -> Violence
            (1, 1): 2      # Not Normal -> Weaponized
        }

        # Extract frames
        frames = self._extract_frames_for_prediction(video_path)

        if use_ensemble:
            # Get predictions from all models
            first_stage_pred = self._predict_first_stage(frames)
            second_stage_pred = self._predict_second_stage(frames)
            weaponized_pred = self._predict_weaponized(frames)

            # Get confidence scores
            first_stage_confidence = self._get_first_stage_confidence(frames, first_stage_pred)
            second_stage_confidence = self._get_second_stage_confidence(frames, second_stage_pred)
            weaponized_confidence = self._get_weaponized_confidence(frames)

            # Ensemble decision logic
            if first_stage_pred == 0:  # If predicted Normal
                if weaponized_confidence > 0.85:  # But weaponized detector disagrees strongly
                    final_pred = 2  # Override to Weaponized
                    final_class = original_classes[final_pred]
                    confidence = weaponized_confidence
                else:
                    final_pred = 0  # Keep as Normal
                    final_class = original_classes[final_pred]
                    confidence = first_stage_confidence
            else:  # If predicted Not Normal
                if weaponized_confidence > 0.8:  # Weaponized detector confident
                    final_pred = 2  # Classify as Weaponized
                    final_class = original_classes[final_pred]
                    confidence = weaponized_confidence
                elif second_stage_pred == 0:  # Second stage says Violence
                    final_pred = 1  # Classify as Violence
                    final_class = original_classes[final_pred]
                    confidence = second_stage_confidence
                else:  # Second stage says Weaponized
                    final_pred = 2  # Classify as Weaponized
                    final_class = original_classes[final_pred]
                    confidence = second_stage_confidence

            # Calculate confidences for all classes using ensemble
            normal_confidence = (1 - weaponized_confidence) * first_stage_confidence

            if second_stage_pred == 0:  # Second stage predicts Violence
                violence_confidence = (1 - normal_confidence) * (1 - weaponized_confidence)
                weaponized_confidence = (1 - normal_confidence) * weaponized_confidence
            else:  # Second stage predicts Weaponized
                violence_confidence = (1 - normal_confidence) * (1 - weaponized_confidence) * 0.3
                weaponized_confidence = (1 - normal_confidence) * (weaponized_confidence +
                                          (1 - weaponized_confidence) * 0.7)

            # Normalize confidences to sum to 1
            total = normal_confidence + violence_confidence + weaponized_confidence
            if total > 0:
                normal_confidence /= total
                violence_confidence /= total
                weaponized_confidence /= total

            return {
                "class": final_class,
                "confidence": confidence,
                "predictions": {
                    original_classes[0]: normal_confidence,
                    original_classes[1]: violence_confidence,
                    original_classes[2]: weaponized_confidence
                },
                "ensemble_scores": {
                    "first_stage": first_stage_confidence,
                    "second_stage": second_stage_confidence,
                    "weaponized_detector": weaponized_confidence
                }
            }
        else:
            # Original two-stage approach
            first_stage_pred = self._predict_first_stage(frames)

            # If Normal, we're done
            if first_stage_pred == 0:
                final_pred = 0
                final_class = original_classes[final_pred]
                confidence = self._get_first_stage_confidence(frames, first_stage_pred)
                return {
                    "class": final_class,
                    "confidence": confidence,
                    "predictions": {
                        original_classes[0]: confidence,
                        original_classes[1]: 0.0,
                        original_classes[2]: 0.0
                    }
                }
            else:
                # If Not Normal, use second stage to distinguish between Violence and Weaponized
                second_stage_pred = self._predict_second_stage(frames)

                # Map to final prediction
                final_pred = class_mapping[(1, second_stage_pred)]
                final_class = original_classes[final_pred]

                # Get confidences
                first_stage_confidence = self._get_first_stage_confidence(frames, first_stage_pred)
                second_stage_confidence = self._get_second_stage_confidence(frames, second_stage_pred)

                # Calculate combined confidence
                confidence = first_stage_confidence * second_stage_confidence

                # Calculate confidences for all classes
                if second_stage_pred == 0:  # Violence
                    violence_confidence = confidence
                    weaponized_confidence = first_stage_confidence * (1 - second_stage_confidence)
                else:  # Weaponized
                    violence_confidence = first_stage_confidence * (1 - second_stage_confidence)
                    weaponized_confidence = confidence

                normal_confidence = 1 - first_stage_confidence

                return {
                    "class": final_class,
                    "confidence": confidence,
                    "predictions": {
                        original_classes[0]: normal_confidence,
                        original_classes[1]: violence_confidence,
                        original_classes[2]: weaponized_confidence
                    }
    }

    def _extract_frames_for_prediction(self, video_path):
        """Extract frames for prediction (similar to dataset, but without label)"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"Error: Could not read frames from {video_path}")
            cap.release()
            return frames

        # Calculate frame indices for even spacing
        if self.frame_count > total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.frame_count, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
                frame = Image.fromarray(frame)
                frame = self.test_transform(frame)
                frames.append(frame)

        cap.release()
        return frames

    def _predict_first_stage(self, frames):
        """Predict using the first stage model (Normal vs Not Normal)"""
        self.first_stage_model.eval()

        # Stack frames
        if len(frames) > 0:
            frames_tensor = torch.stack(frames).to(self.device)

            with torch.no_grad():
                outputs = self.first_stage_model(frames_tensor)
                _, predicted = torch.max(outputs, 1)

            # Return most frequent prediction, assuming it's the most reliable
            return torch.mode(predicted).values.item()
        else:
            # Handle case where no frames were extracted
            return 0 # Default to normal if no frames

    def _predict_second_stage(self, frames):
        """Predict using the second stage model (Violence vs Weaponized)"""
        self.second_stage_model.eval()

        # Stack frames into a sequence
        if len(frames) > 0:
            frames_tensor = torch.stack(frames).unsqueeze(0).to(self.device) # Add batch dimension

            with torch.no_grad():
                outputs = self.second_stage_model(frames_tensor)
                _, predicted = torch.max(outputs, 1)

            return predicted.item()
        else:
            # Handle no frames extracted by defaulting to violence.
            return 0

    def _predict_weaponized(self, frames):
        """Predict using the weaponized detector model (Normal vs Weaponized)"""
        self.weaponized_detector.eval()

        # Stack frames
        if len(frames) > 0:
            frames_tensor = torch.stack(frames).to(self.device)
            with torch.no_grad():
                outputs = self.weaponized_detector(frames_tensor)
                _, predicted = torch.max(outputs, 1)
            return torch.mode(predicted).values.item()
        else:
            # Handle case where no frames were extracted
            return 0 #Default to Normal if no frames

    def _get_first_stage_confidence(self, frames, prediction):
        """Get confidence score for the first stage prediction"""
        self.first_stage_model.eval()
        if len(frames) > 0:
            frames_tensor = torch.stack(frames).to(self.device)
            with torch.no_grad():
                outputs = self.first_stage_model(frames_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                if prediction == 0:
                    confidence = probabilities[:, 0].mean().item() # Normal
                else:
                    confidence = probabilities[:, 1].mean().item() #Not Normal
            return confidence
        else:
            return 0.0

    def _get_second_stage_confidence(self, frames, prediction):
        """Get confidence score for the second stage prediction"""
        self.second_stage_model.eval()
        if len(frames) > 0:
            frames_tensor = torch.stack(frames).unsqueeze(0).to(self.device) # Add batch dimension

            with torch.no_grad():
                outputs = self.second_stage_model(frames_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                if prediction == 0:
                    confidence = probabilities[:, 0].mean().item()  # Violence
                else:
                    confidence = probabilities[:, 1].mean().item()  # Weaponized
            return confidence
        else:
            return 0.0

    def _get_weaponized_confidence(self, frames):
        """Get confidence score from the weaponized detector"""
        self.weaponized_detector.eval()

        if len(frames) > 0:
            frames_tensor = torch.stack(frames).to(self.device)

            with torch.no_grad():
                outputs = self.weaponized_detector(frames_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                weaponized_confidence = probabilities[:, 1].mean().item()  # Confidence for Weaponized
            return weaponized_confidence
        else:
            return 0.0