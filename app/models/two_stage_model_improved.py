# two_stage_model_improved.py
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
import math
import torch
import random
from torch.cuda.amp import autocast, GradScaler

class VideoDataset(Dataset):
    def __init__(self, video_data, transform=None, frame_count=10, split='train', temporal_features=True):
        """
        Args:
            video_data (dict): Dictionary of video paths returned by prepare_classification_dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            frame_count (int): Number of frames to extract per video.
            split (str): 'train', 'test', or 'val'.
            temporal_features (bool): Whether to return temporal features (sequence of frames) or averaged frames.
        """
        self.video_paths = []
        self.labels = []
        self.transform = transform
        self.frame_count = frame_count
        self.temporal_features = temporal_features

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

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self._extract_frames(video_path, self.frame_count)

        if len(frames) > 0:
            # Process the extracted frames
            processed_frames = self._process_frames(frames)

            if self.temporal_features:
                # Return sequence of frames for temporal processing
                # If we don't have enough frames, repeat the last frame
                if processed_frames.size(0) < self.frame_count:
                    last_frame = processed_frames[-1].unsqueeze(0)
                    padding = last_frame.repeat(self.frame_count - processed_frames.size(0), 1, 1, 1)
                    processed_frames = torch.cat([processed_frames, padding], dim=0)
                # If we have too many frames, take frames at regular intervals
                elif processed_frames.size(0) > self.frame_count:
                    indices = torch.linspace(0, processed_frames.size(0) - 1, self.frame_count).long()
                    processed_frames = processed_frames[indices]

                return processed_frames, self.class_to_idx[label]
            else:
                # Average the features across frames for non-temporal processing
                avg_frame = torch.mean(processed_frames, dim=0)
                return avg_frame, self.class_to_idx[label]
        else:
            print(f"Warning: No frames extracted from {video_path}")
            if self.temporal_features:
                # Return zero tensor of shape [frame_count, channels, height, width]
                return torch.zeros((self.frame_count, 3, 224, 224)), self.class_to_idx[label]
            else:
                return torch.zeros((3, 224, 224)), self.class_to_idx[label]

    def _extract_frames(self, video_path, frame_count):
        """Extract frames from a video file with improved error handling"""
        frames = []
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                print(f"Error: Video file {video_path} does not exist")
                return frames
                
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return frames

            # Test if video is readable
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print(f"Error: Video file {video_path} appears to be corrupted")
                return frames
                
            # Reset to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Get video properties
            frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if frame_count_total <= 0:
                print(f"Error: Video {video_path} has invalid frame count: {frame_count_total}")
                return frames

            # Calculate step size for uniform sampling across the video
            step = max(1, frame_count_total // frame_count)
            
            # Use uniform sampling with a small random offset for training data
            if self.split == 'train':
                # Add small random offsets for better generalization during training
                random_offsets = [random.randint(-5, 5) for _ in range(min(frame_count, frame_count_total // step))]
                frame_indices = [max(0, min(frame_count_total - 1, (i * step) + (random_offsets[i] if i < len(random_offsets) else 0))) 
                                for i in range(min(frame_count, frame_count_total // step))]
            else:
                # For validation/test, use uniform sampling without randomness
                frame_indices = [i * step for i in range(min(frame_count, frame_count_total // step))]

            for idx in frame_indices:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frames.append(frame)
                    else:
                        print(f"Warning: Could not read frame {idx} from {video_path}")
                except Exception as e:
                    print(f"Error reading frame {idx} from {video_path}: {e}")

            # Release the video capture object
            cap.release()

        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            import traceback
            traceback.print_exc()

        return frames

    def _process_frames(self, frames):
        """Process extracted frames: convert to RGB, apply transforms, and stack"""
        processed_frames = []
        for frame in frames:
            # Convert to RGB (OpenCV loads as BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            if self.transform:
                frame = self.transform(frame)
            processed_frames.append(frame)

        # Stack frames along a new dimension
        if len(processed_frames) > 0:
            stacked_frames = torch.stack(processed_frames)
            return stacked_frames
        else:
            return torch.tensor([])


# Improved Temporal CNN with Attention Mechanism
class EnhancedTemporalCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.6):
        super(EnhancedTemporalCNN, self).__init__()
        # Define the model architecture
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Feature extraction base (using EfficientNet as backbone for better feature extraction)
        self.base_model = models.efficientnet_b2(pretrained=True)
        # Remove classifier
        self.base_features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Get feature dimensions for EfficientNet-B2
        self.feature_dim = 1408  # EfficientNet-B2 feature dimension

        # Temporal modeling with 3D convolutions
        self.temporal_conv1 = nn.Conv3d(self.feature_dim, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.temporal_bn1 = nn.BatchNorm3d(512)
        self.temporal_conv2 = nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.temporal_bn2 = nn.BatchNorm3d(256)
        # Feature reduction layer
        self.feature_selection = nn.Conv2d(self.feature_dim, self.feature_dim // 2, kernel_size=1)
        self.feature_dim = self.feature_dim // 2  # Update feature dimension
        
        # Attention mechanism for better temporal feature extraction
        self.attention = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(64, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer for classification
        self.fc = nn.Linear(256, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize temporal convolution layers
        for m in [self.temporal_conv1, self.temporal_conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Initialize batch normalization layers
        for m in [self.temporal_bn1, self.temporal_bn2]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        # Initialize fully connected layer
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x shape: [batch_size, timesteps, channels, height, width]
        batch_size, timesteps, c, h, w = x.size()

        # Process each frame through the base model
        x = x.view(batch_size * timesteps, c, h, w)
        x = self.base_features(x)

        # Apply feature selection
        x = x.view(batch_size * timesteps, self.feature_dim, 1, 1)
        x = self.feature_selection(x)
        x = F.relu(x)

        # Reshape for temporal processing
        x = x.view(batch_size, timesteps, self.feature_dim, 1, 1)
        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, channels, timesteps, height, width]

        # Apply temporal convolutions
        x = F.relu(self.temporal_bn1(self.temporal_conv1(x)))
        x = F.relu(self.temporal_bn2(self.temporal_conv2(x)))
        
        # Apply attention mechanism
        attn = self.attention(x)
        x = x * attn  # Element-wise multiplication

        # Global average pooling
        x = self.pool(x)
        x = x.view(batch_size, -1)

        # Apply dropout and final classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


class TwoStageModel:
    def __init__(self, video_data, model_path=None, frame_count=10, dropout_rate=0.8, 
                 use_weighted_sampler=True, class_weight_epsilon=5):
        """Initialize with improved parameters"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.frame_count = frame_count
        self.use_weighted_sampler = use_weighted_sampler
        self.class_weight_epsilon = class_weight_epsilon

        # Enhanced transforms with more augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.7),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.4),  # New
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15))  # Increased from 0.2
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize datasets
        self._prepare_first_stage_data(video_data)
        self._prepare_second_stage_data(video_data)

        # Initialize models with higher dropout
        self.first_stage_model = self._create_model(num_classes=2, dropout_rate=dropout_rate)
        self.second_stage_model = EnhancedTemporalCNN(num_classes=2, dropout_rate=dropout_rate)
        
        self.first_stage_model = self.first_stage_model.to(self.device)
        self.second_stage_model = self.second_stage_model.to(self.device)
        self.scaler = GradScaler()

    def _create_model(self, num_classes, dropout_rate=0.7):
        """Create model with configurable dropout"""
        model = models.efficientnet_b2(pretrained=True)
        
        # Freeze early layers
        for name, param in model.named_parameters():
            if 'features.0' in name or 'features.1' in name:
                param.requires_grad = False

        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2, inplace=True),
            nn.Linear(512, num_classes)
        )
        return model

    def _prepare_first_stage_data(self, video_data):
        """Prepare data with improved class weighting"""
        binary_video_data = {'train': {}, 'test': {}}
        binary_video_data['train']['Normal'] = video_data['train']['Normal']
        binary_video_data['test']['Normal'] = video_data['test']['Normal']
        binary_video_data['train']['Not_Normal'] = []
        binary_video_data['test']['Not_Normal'] = []

        for split in ['train', 'test']:
            for class_name in ['Violence', 'Weaponized']:
                if class_name in video_data[split]:
                    binary_video_data[split]['Not_Normal'].extend(video_data[split][class_name])

        # Initialize datasets
        self.first_stage_train_dataset = VideoDataset(
            binary_video_data, transform=self.train_transform,
            frame_count=self.frame_count, split='train', temporal_features=False
        )

        self.first_stage_test_dataset = VideoDataset(
            binary_video_data, transform=self.test_transform,
            frame_count=self.frame_count, split='test', temporal_features=False
        )

        # Improved weighted sampler with epsilon
        if self.use_weighted_sampler:
            class_counts = Counter([self.first_stage_train_dataset.class_to_idx[label]
                                  for label in self.first_stage_train_dataset.labels])
            class_weights = {class_idx: 1.0 / (count + self.class_weight_epsilon) 
                           for class_idx, count in class_counts.items()}
            sample_weights = [class_weights[self.first_stage_train_dataset.class_to_idx[label]]
                             for label in self.first_stage_train_dataset.labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        self.first_stage_train_loader = DataLoader(
            self.first_stage_train_dataset, batch_size=32, shuffle=shuffle,
            sampler=sampler, num_workers=4, pin_memory=True  # Increased workers
        )

        self.first_stage_test_loader = DataLoader(
            self.first_stage_test_dataset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True
        )

    def _prepare_second_stage_data(self, video_data):
        """Prepare data for the second stage: Violence vs Weaponized"""
        # Create a subset of video_data with only Violence and Weaponized classes
        subset_video_data = {'train': {}, 'test': {}}

        for split in ['train', 'test']:
            for class_name in ['Violence', 'Weaponized']:
                if class_name in video_data[split]:
                    subset_video_data[split][class_name] = video_data[split][class_name]

        # Initialize datasets
        self.second_stage_train_dataset = VideoDataset(
            subset_video_data, transform=self.train_transform,
            frame_count=self.frame_count, split='train', temporal_features=True
        )

        self.second_stage_test_dataset = VideoDataset(
            subset_video_data, transform=self.test_transform,
            frame_count=self.frame_count, split='test', temporal_features=True
        )

        # Create weighted sampler for training data
        if self.use_weighted_sampler:
            class_counts = Counter([self.second_stage_train_dataset.class_to_idx[label]
                                   for label in self.second_stage_train_dataset.labels])
            class_weights = {class_idx: 1.0 / count for class_idx, count in class_counts.items()}
            sample_weights = [class_weights[self.second_stage_train_dataset.class_to_idx[label]]
                             for label in self.second_stage_train_dataset.labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        # Initialize DataLoaders with worker_init_fn for stability
        self.second_stage_train_loader = DataLoader(
            self.second_stage_train_dataset, batch_size=16, shuffle=shuffle,
            sampler=sampler, num_workers=0, pin_memory=True,
            worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
        )

        self.second_stage_test_loader = DataLoader(
            self.second_stage_test_dataset, batch_size=16, shuffle=False,
            num_workers=0, pin_memory=True,
            worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
        )

    def _train_epoch(self, epoch, total_epochs, model, train_loader, val_loader, 
                     optimizer, criterion, history, stage_name, scheduler=None):
        """Helper method to train one epoch for any model
        
        Args:
            epoch (int): Current epoch
            total_epochs (int): Total number of epochs
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            optimizer (Optimizer): Optimizer
            criterion (Loss): Loss function
            history (dict): History dictionary to update
            stage_name (str): Name of the stage for logging
            scheduler (LRScheduler, optional): Learning rate scheduler
            
        Returns:
            float: Validation accuracy
        """
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if 'learning_rates' in history:
            history['learning_rates'].append(current_lr)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            # Backward pass and optimize with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Clip gradients to prevent exploding gradients
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Update learning rate if scheduler is provided
            if scheduler:
                scheduler.step()
                
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"{stage_name} - Epoch {epoch + 1}/{total_epochs} | "
                      f"Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        val_loss, val_acc = self._validate(model, val_loader, criterion)
        
        # Update history
        if 'train_loss' in history:
            history['train_loss'].append(train_loss)
        if 'train_acc' in history:
            history['train_acc'].append(train_acc)
        if 'val_loss' in history:
            history['val_loss'].append(val_loss)
        if 'val_acc' in history:
            history['val_acc'].append(val_acc)
        
        # Print statistics
        print(f"{stage_name} - Epoch {epoch + 1}/{total_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"LR: {current_lr:.6f}")
        
        return val_acc
    

    def _validate(self, model, val_loader, criterion):
        """Validate the model on validation data"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        return val_loss, val_acc    
            
    def train_first_stage(self, epochs=30, learning_rate=0.0003, weight_decay=1e-3, 
                    patience=8, label_smoothing=0.1, mixup_alpha=0.2):
        """Improved first stage training with new parameters"""
        # Compute class weights
        # if class_weight_epsilon is None:
        #     class_weight_epsilon = self.class_weight_epsilon

        y_train = np.array([self.first_stage_train_dataset.class_to_idx[label]
                        for label in self.first_stage_train_dataset.labels])
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print(f"First stage class weights: {class_weights}")

        history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [], 
            'learning_rates': []
        }

        # Warm-up phase
        print("First stage warm-up: training only classifier")
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        for param in self.first_stage_model.classifier.parameters():
            param.requires_grad = True

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.first_stage_model.parameters()),
            lr=learning_rate/10,  # Lower warm-up LR
            weight_decay=weight_decay
        )

        warmup_epochs = max(5, epochs // 6)
        for epoch in range(warmup_epochs):
            self._train_epoch(epoch, warmup_epochs, self.first_stage_model,
                            self.first_stage_train_loader, self.first_stage_test_loader,
                            optimizer, criterion, history, "First Stage Warm-up")

        # Main training
        print("First stage main training: unfreezing all layers")
        for param in self.first_stage_model.parameters():
            param.requires_grad = True

        # Step 8: Use different learning rates for different parts of the model
        params = [
            {'params': [p for n, p in self.first_stage_model.named_parameters() if 'classifier' not in n], 'lr': learning_rate/10},
            {'params': self.first_stage_model.classifier.parameters(), 'lr': learning_rate}
        ]
        optimizer = optim.AdamW(params, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate*3,  # Reduced peak LR from *10 to *5
            epochs=epochs - warmup_epochs,
            steps_per_epoch=len(self.first_stage_train_loader),
            pct_start=0.2,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )

        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0.0
        no_improve_counter = 0
        best_model_path = os.path.join('models', 'first_stage_best.pth')

        for epoch in range(epochs - warmup_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

            self.first_stage_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, labels) in enumerate(self.first_stage_train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Step 7: Apply mixup augmentation
                if mixup_alpha > 0 and np.random.random() < 0.5:  # Apply mixup 50% of the time
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    index = torch.randperm(inputs.size(0)).to(self.device)
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                    labels_a, labels_b = labels, labels[index]
                    
                    optimizer.zero_grad()
                    with autocast():
                        outputs = self.first_stage_model(mixed_inputs)
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    
                    # For accuracy calculation, use the primary labels
                    _, predicted = torch.max(outputs, 1)
                else:
                    optimizer.zero_grad()
                    with autocast():
                        outputs = self.first_stage_model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs, 1)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.first_stage_model.parameters(), max_norm=0.5)  # Changed from 1.0 to 0.5
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()

                train_loss += loss.item() * inputs.size(0)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"First Stage - Epoch {epoch + 1}/{epochs - warmup_epochs} | "
                        f"Batch {batch_idx + 1}/{len(self.first_stage_train_loader)} | "
                        f"Loss: {loss.item():.4f}")

            train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            val_loss, val_acc = self._validate(
                self.first_stage_model, self.first_stage_test_loader, criterion
            )

            if val_acc > best_val_acc:
                print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                no_improve_counter = 0
                torch.save(self.first_stage_model.state_dict(), best_model_path)
            else:
                no_improve_counter += 1

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"First Stage - Epoch {epoch + 1}/{epochs - warmup_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.6f}")

            if no_improve_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        print("Loading best first stage model")
        self.first_stage_model.load_state_dict(torch.load(best_model_path))
        return history
    def train_second_stage(self, epochs=80, learning_rate=0.0005, weight_decay=3e-4, patience=20,
                      label_smoothing=0.05, mixup_alpha=0.2):
        """Train the second stage model (Violence vs Weaponized) with enhanced training techniques"""
        print("Training second stage model: Violence vs Weaponized")

        # Compute class weights for loss function
        y_train = np.array([self.second_stage_train_dataset.class_to_idx[label]
                        for label in self.second_stage_train_dataset.labels])
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print(f"Second stage class weights: {class_weights}")

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(
            self.second_stage_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Use a 1cycle policy for learning rate scheduling
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,  # Peak learning rate
            epochs=epochs,
            steps_per_epoch=len(self.second_stage_train_loader),
            pct_start=0.3,  # Spend 30% of time warming up
            anneal_strategy='cos',
            div_factor=25.0,  # initial_lr = max_lr/25
            final_div_factor=10000.0  # min_lr = initial_lr/10000
        )

        # Initialize tracking variables
        best_val_acc = 0.0
        no_improve_counter = 0
        best_model_path = os.path.join('models', 'second_stage_best.pth')


        # Add this code block for resuming training
        start_epoch = 0
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint)
            self.second_stage_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            print(f"Resuming training from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
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
                try:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Apply mixup augmentation
                    if mixup_alpha > 0 and np.random.random() < 0.8:  # Apply mixup 80% of the time
                        lam = np.random.beta(mixup_alpha, mixup_alpha)
                        index = torch.randperm(inputs.size(0)).to(self.device)
                        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                        labels_a, labels_b = labels, labels[index]

                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        # Forward pass with mixed inputs using mixed precision
                        with autocast():
                            outputs = self.second_stage_model(mixed_inputs)
                            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

                        # For accuracy calculation, we'll use the primary labels
                        _, predicted = torch.max(outputs, 1)
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()
                    else:
                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        # Forward pass with mixed precision
                        with autocast():
                            outputs = self.second_stage_model(inputs)
                            loss = criterion(outputs, labels)

                        # Track statistics
                        _, predicted = torch.max(outputs, 1)
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()

                    # Backward pass and optimize with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Clip gradients to prevent exploding gradients
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.second_stage_model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    # Update learning rate
                    scheduler.step()

                    # Track statistics
                    train_loss += loss.item() * inputs.size(0)

                    # Print progress every 5 batches (since batch size is smaller)
                    if (batch_idx + 1) % 5 == 0:
                        print(f"Second Stage - Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(self.second_stage_train_loader)} | "
                            f"Loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    # Skip problematic batch and continue with next one
                    continue

            # Check if any training was done
            if train_total == 0:
                print("No successful training batches in this epoch. Skipping evaluation.")
                continue

            # Calculate average training loss and accuracy
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation phase
            try:
                val_loss, val_acc = self._validate(
                    self.second_stage_model, self.second_stage_test_loader, criterion
                )
            except Exception as e:
                print(f"Error during validation: {e}")
                # Use previous validation metrics if validation fails
                val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
                val_acc = history['val_acc'][-1] if history['val_acc'] else 0.0
                print(f"Using previous metrics: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            # Check if this is the best model
            if val_acc > best_val_acc:
                print(f"Second Stage - Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                no_improve_counter = 0
                torch.save(self.second_stage_model.state_dict(), best_model_path)
                print(f"Saved best second stage model at epoch {epoch + 1}")
            else:
                no_improve_counter += 1

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print statistics
            print(f"Second Stage - Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.6f}")

            # Early stopping check
            if no_improve_counter >= patience:
                print(f"Second Stage - Early stopping triggered at epoch {epoch + 1}")
                break

        # Load the best model
        print(f"Loading best second stage model")
        self.second_stage_model.load_state_dict(torch.load(best_model_path))
        self.second_stage_model.eval()

        return history, best_model_path
    def train(self, epochs_first=35, epochs_second=90, learning_rate_first=0.0004, 
        learning_rate_second=0.0006, weight_decay=4e-4, patience_first=12, 
        patience_second=25, label_smoothing=0.05, mixup_alpha=0.3):
        """Train both stages of the model with improved parameters"""
        try:
            print("Training first stage model: Normal vs Not Normal")
            first_stage_history = self.train_first_stage(
                epochs=epochs_first,
                learning_rate=learning_rate_first,
                weight_decay=weight_decay,
                patience=patience_first,
                label_smoothing=label_smoothing
            )
            
            print("Training second stage model: Violence vs Weaponized")
            second_stage_history, second_stage_path = self.train_second_stage(
                epochs=epochs_second,
                learning_rate=learning_rate_second,
                weight_decay=weight_decay,
                patience=patience_second,
                label_smoothing=label_smoothing,
                mixup_alpha=mixup_alpha
            )
            
            # Get the path to the saved first stage model
            first_stage_path = os.path.join('models', 'first_stage_best.pth')
            
            # Return training history and model paths
            history = {
                'first_stage': first_stage_history,
                'second_stage': second_stage_history
            }
            
            model_paths = {
                'first_stage': first_stage_path,
                'second_stage': second_stage_path
            }
            
            return history, model_paths
        except Exception as e:
            print(f"ERROR: Training process encountered an exception: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to return whatever we have
            history = {}
            model_paths = {}
            
            if 'first_stage_history' in locals():
                history['first_stage'] = first_stage_history
                model_paths['first_stage'] = os.path.join('models', 'first_stage_best.pth')
            
            if 'second_stage_history' in locals() and 'second_stage_path' in locals():
                history['second_stage'] = second_stage_history
                model_paths['second_stage'] = second_stage_path
            
            return history, model_paths      
    def evaluate(self, verbose=True):
        """Evaluate the two-stage model on the test set
        
        Args:
            verbose (bool): Whether to print detailed results. Default is True.
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        try:
            print("Evaluating two-stage model on test set...")
            
            # Initialize metrics
            all_preds = []
            all_labels = []
            
            # Set models to evaluation mode
            self.first_stage_model.eval()
            self.second_stage_model.eval()
            
            # First stage evaluation: Normal vs Not Normal
            normal_correct = 0
            normal_total = 0
            not_normal_correct = 0
            not_normal_total = 0
            
            with torch.no_grad():
                for inputs, labels in self.first_stage_test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.first_stage_model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Count correct predictions for each class
                    normal_mask = (labels == 0)  # Normal class
                    normal_total += normal_mask.sum().item()
                    normal_correct += ((predicted == labels) & normal_mask).sum().item()
                    
                    not_normal_mask = (labels == 1)  # Not Normal class
                    not_normal_total += not_normal_mask.sum().item()
                    not_normal_correct += ((predicted == labels) & not_normal_mask).sum().item()
            
            # Calculate first stage accuracy
            normal_acc = normal_correct / normal_total if normal_total > 0 else 0
            not_normal_acc = not_normal_correct / not_normal_total if not_normal_total > 0 else 0
            first_stage_acc = (normal_correct + not_normal_correct) / (normal_total + not_normal_total)
            
            if verbose:
                print(f"First Stage Results:")
                print(f"  Normal Class:     {normal_correct}/{normal_total} ({normal_acc:.4f})")
                print(f"  Not Normal Class: {not_normal_correct}/{not_normal_total} ({not_normal_acc:.4f})")
                print(f"  Overall Accuracy: {first_stage_acc:.4f}")
            
            # Second stage evaluation: Violence vs Weaponized
            violence_correct = 0
            violence_total = 0
            weaponized_correct = 0
            weaponized_total = 0
            
            with torch.no_grad():
                for inputs, labels in self.second_stage_test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.second_stage_model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Count correct predictions for each class
                    violence_mask = (labels == 0)  # Violence class
                    violence_total += violence_mask.sum().item()
                    violence_correct += ((predicted == labels) & violence_mask).sum().item()
                    
                    weaponized_mask = (labels == 1)  # Weaponized class
                    weaponized_total += weaponized_mask.sum().item()
                    weaponized_correct += ((predicted == labels) & weaponized_mask).sum().item()
            
            # Calculate second stage accuracy
            violence_acc = violence_correct / violence_total if violence_total > 0 else 0
            weaponized_acc = weaponized_correct / weaponized_total if weaponized_total > 0 else 0
            second_stage_acc = (violence_correct + weaponized_correct) / (violence_total + weaponized_total)
            
            if verbose:
                print(f"Second Stage Results:")
                print(f"  Violence Class:    {violence_correct}/{violence_total} ({violence_acc:.4f})")
                print(f"  Weaponized Class:  {weaponized_correct}/{weaponized_total} ({weaponized_acc:.4f})")
                print(f"  Overall Accuracy:  {second_stage_acc:.4f}")
            
            # Calculate overall accuracy
            total_samples = normal_total + not_normal_total
            correct_samples = normal_correct + (not_normal_correct * second_stage_acc)
            overall_acc = correct_samples / total_samples
            
            if verbose:
                print(f"Overall Model Results:")
                print(f"  Accuracy: {overall_acc:.4f}")
            
            # Create confusion matrix
            # This is a simplified approximation since we don't have the full end-to-end predictions
            confusion_matrix = torch.zeros((3, 3), dtype=torch.long)
            
            # Populate confusion matrix
            # Normal class (correctly classified)
            confusion_matrix[0, 0] = normal_correct
            # Normal class (misclassified)
            confusion_matrix[0, 1] = int((normal_total - normal_correct) * violence_acc)  # As Violence
            confusion_matrix[0, 2] = int((normal_total - normal_correct) * weaponized_acc)  # As Weaponized
            
            # Violence class
            confusion_matrix[1, 0] = int(violence_total * (1 - not_normal_acc))  # As Normal
            confusion_matrix[1, 1] = int(violence_total * not_normal_acc * violence_acc)  # As Violence
            confusion_matrix[1, 2] = int(violence_total * not_normal_acc * (1 - violence_acc))  # As Weaponized
            
            # Weaponized class
            confusion_matrix[2, 0] = int(weaponized_total * (1 - not_normal_acc))  # As Normal
            confusion_matrix[2, 1] = int(weaponized_total * not_normal_acc * (1 - weaponized_acc))  # As Violence
            confusion_matrix[2, 2] = int(weaponized_total * not_normal_acc * weaponized_acc)  # As Weaponized
            
            if verbose:
                print("\nConfusion Matrix:")
                classes = ['Normal', 'Violence', 'Weaponized']
                print("            " + " ".join([f"{name:>10}" for name in classes]))
                for i, name in enumerate(classes):
                    print(f"{name:10}" + " ".join([f"{confusion_matrix[i, j]:10d}" for j in range(len(classes))]))
            
            # Return metrics
            metrics = {
                'accuracy': overall_acc,
                'first_stage_acc': first_stage_acc,
                'second_stage_acc': second_stage_acc,
                'class_acc': {
                    'Normal': normal_acc,
                    'Violence': violence_acc,
                    'Weaponized': weaponized_acc
                },
                'confusion_matrix': confusion_matrix.cpu().numpy()
            }
            
            return metrics  
        except Exception as e:
            print(f"ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            
            # Return partial metrics if possible
            return {
                'error': str(e),
                'accuracy': 0.0,
                'first_stage_acc': 0.0,
                'second_stage_acc': 0.0
            }           
