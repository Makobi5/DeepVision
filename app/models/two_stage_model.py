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

class VideoDataset(Dataset):
    def __init__(self, video_data, transform=None, frame_count=8, split='train', temporal_features=True):
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
                # If we have too many frames, take the first frame_count frames
                elif processed_frames.size(0) > self.frame_count:
                    processed_frames = processed_frames[:self.frame_count]

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
        """Extract frames from a video file"""
        frames = []
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return frames

            # Get video properties
            frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if frame_count_total == 0:
                print(f"Error: Video {video_path} has 0 frames")
                return frames

            # Calculate step size to evenly sample frames
            step = max(1, frame_count_total // frame_count)

            # Extract frames
            frame_indices = [i * step for i in range(min(frame_count, frame_count_total // step))]

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    print(f"Warning: Could not read frame {idx} from {video_path}")

            # Release the video capture object
            cap.release()

        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")

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


class EnhancedTemporalCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(EnhancedTemporalCNN, self).__init__()
        # Define the model architecture
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Feature extraction base (using pretrained ResNet as the backbone)
        resnet = models.resnet50(pretrained=True)
        self.base_model = nn.Sequential(*list(resnet.children())[:-2])  # Remove avg pool and fc layers

        # Temporal modeling with 3D convolutions
        self.temporal_conv1 = nn.Conv3d(2048, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.temporal_bn1 = nn.BatchNorm3d(512)
        self.temporal_conv2 = nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.temporal_bn2 = nn.BatchNorm3d(256)

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
        x = self.base_model(x)

        # Reshape for temporal processing
        _, c, h, w = x.size()
        x = x.view(batch_size, timesteps, c, h, w)
        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, channels, timesteps, height, width]

        # Apply temporal convolutions
        x = F.relu(self.temporal_bn1(self.temporal_conv1(x)))
        x = F.relu(self.temporal_bn2(self.temporal_conv2(x)))

        # Global average pooling
        x = self.pool(x)
        x = x.view(batch_size, -1)

        # Apply dropout and final classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


class TwoStageModel:
    def __init__(self, video_data, model_path=None, frame_count=8, dropout_rate=0.5, use_weighted_sampler=True):
        """Initialize the two-stage model for video classification

        Args:
            video_data (dict): Dictionary containing video paths for training and testing.
            model_path (dict, optional): Dictionary containing paths to pre-trained weights. Default is None.
            frame_count (int): Number of frames to extract per video. Default is 8.
            dropout_rate (float): Dropout rate for the models. Default is 0.5.
            use_weighted_sampler (bool): Whether to use weighted sampler for imbalanced classes. Default is True.
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Set parameters
        self.frame_count = frame_count
        self.use_weighted_sampler = use_weighted_sampler

        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        # Initialize models
        self.first_stage_model = self._create_model(num_classes=2)  # Normal vs Not Normal
        # Use the enhanced temporal CNN for second stage
        self.second_stage_model = EnhancedTemporalCNN(num_classes=2, dropout_rate=dropout_rate)

        # Move models to device
        self.first_stage_model = self.first_stage_model.to(self.device)
        self.second_stage_model = self.second_stage_model.to(self.device)

        # Load pre-trained weights if provided
        if model_path and isinstance(model_path, dict):
            if 'first_stage' in model_path and os.path.exists(model_path['first_stage']):
                self.first_stage_model.load_state_dict(torch.load(model_path['first_stage'], map_location=self.device))
                print(f"Loaded first stage weights from {model_path['first_stage']}")

            if 'second_stage' in model_path and os.path.exists(model_path['second_stage']):
                self.second_stage_model.load_state_dict(torch.load(model_path['second_stage'], map_location=self.device))
                print(f"Loaded second stage weights from {model_path['second_stage']}")

    def _create_model(self, num_classes):
        """Create a model for classification based on ResNet50"""
        # Load pre-trained ResNet50
        model = models.resnet50(pretrained=True)

        # Freeze early layers to prevent overfitting
        for param in list(model.parameters())[:-20]:  # Freeze all but the last few layers
            param.requires_grad = False

        # Replace the final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(num_features, num_classes)
        )

        return model

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
            frame_count=self.frame_count, split='train', temporal_features=False
        )

        self.first_stage_test_dataset = VideoDataset(
            binary_video_data, transform=self.test_transform,
            frame_count=self.frame_count, split='test', temporal_features=False
        )

        # Create weighted sampler for training data
        if self.use_weighted_sampler:
            class_counts = Counter([self.first_stage_train_dataset.class_to_idx[label]
                                   for label in self.first_stage_train_dataset.labels])
            class_weights = {class_idx: 1.0 / count for class_idx, count in class_counts.items()}
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

        # Initialize DataLoaders with a smaller batch size for 3D models
        self.second_stage_train_loader = DataLoader(
            self.second_stage_train_dataset, batch_size=16, shuffle=shuffle,
            sampler=sampler, num_workers=4, pin_memory=True
        )

        self.second_stage_test_loader = DataLoader(
            self.second_stage_test_dataset, batch_size=16, shuffle=False,
            num_workers=4, pin_memory=True
        )

    def _extract_frames_for_prediction(self, video_path):
        """Extract frames from a video for prediction"""
        frames = self.first_stage_train_dataset._extract_frames(video_path, self.frame_count)

        if len(frames) == 0:
            print(f"Warning: No frames extracted from {video_path} for prediction")
            # Return empty tensor of appropriate shape
            return torch.zeros((self.frame_count, 3, 224, 224)).to(self.device)

        # Process frames
        processed_frames = self.first_stage_train_dataset._process_frames(frames)

        # Handle if we don't have enough frames
        if processed_frames.size(0) < self.frame_count:
            last_frame = processed_frames[-1].unsqueeze(0)
            padding = last_frame.repeat(self.frame_count - processed_frames.size(0), 1, 1, 1)
            processed_frames = torch.cat([processed_frames, padding], dim=0)
        elif processed_frames.size(0) > self.frame_count:
            processed_frames = processed_frames[:self.frame_count]

        # Move to device
        processed_frames = processed_frames.to(self.device)

        return processed_frames

    def _predict_first_stage(self, frames):
        """Make prediction using the first stage model"""
        self.first_stage_model.eval()

        with torch.no_grad():
            # For first stage model, we use the average of frames
            if frames.dim() == 4:  # [timesteps, channels, height, width]
                avg_frame = torch.mean(frames, dim=0).unsqueeze(0)  # Add batch dimension
            else:  # Already batched
                avg_frame = torch.mean(frames, dim=1).unsqueeze(0)

            # Forward pass
            outputs = self.first_stage_model(avg_frame)

            # Get prediction (0: Normal, 1: Not Normal)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()

    def _predict_second_stage(self, frames):
        """Make prediction using the second stage model"""
        self.second_stage_model.eval()

        with torch.no_grad():
            # Add batch dimension if not already present
            if frames.dim() == 4:  # [timesteps, channels, height, width]
                frames = frames.unsqueeze(0)  # Add batch dimension

            # Forward pass
            outputs = self.second_stage_model(frames)

            # Get prediction (0: Violence, 1: Weaponized)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()

    def train_first_stage(self, epochs=30, learning_rate=0.0003, weight_decay=3e-4, patience=10):
        """Train the first stage model (Normal vs Not Normal)

        Args:
            epochs (int): Number of epochs for training. Default is 30.
            learning_rate (float): Learning rate for optimizer. Default is 0.0003.
            weight_decay (float): Weight decay for regularization. Default is 3e-4.
            patience (int): Early stopping patience. Default is 10.

        Returns:
            dict: Training history containing loss, accuracy and learning rates
        """
        print("Training first stage model: Normal vs Not Normal")

        # Compute class weights for loss function
        y_train = np.array([self.first_stage_train_dataset.class_to_idx[label]
                           for label in self.first_stage_train_dataset.labels])
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print(f"First stage class weights: {class_weights}")

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            self.first_stage_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Use a 1cycle policy for learning rate scheduling
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=len(self.first_stage_train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )

        # Initialize tracking variables
        best_val_acc = 0.0
        no_improve_counter = 0
        best_model_path = os.path.join('models', 'first_stage_best.pth')

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
                optimizer.step()

                # Update learning rate
                scheduler.step()

                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"First Stage - Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(self.first_stage_train_loader)} | "
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

            # Check if this is the best model
            if val_acc > best_val_acc:
                print(f"First Stage - Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                no_improve_counter = 0
                torch.save(self.first_stage_model.state_dict(), best_model_path)
                print(f"Saved best first stage model at epoch {epoch + 1}")
            else:
                no_improve_counter += 1

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print statistics
            print(f"First Stage - Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")

            # Early stopping check
            if no_improve_counter >= patience:
                print(f"First Stage - Early stopping triggered at epoch {epoch + 1}")
                break

        # Load the best model
        print(f"Loading best first stage model")
        self.first_stage_model.load_state_dict(torch.load(best_model_path))
        self.first_stage_model.eval()

        return history

    def train_second_stage(self, epochs=80, learning_rate=0.0005, weight_decay=3e-4, patience=20,
                          label_smoothing=0.05, mixup_alpha=0.2):
        """Train the second stage model (Violence vs Weaponized) with enhanced training techniques

        Args:
            epochs (int): Number of epochs for training. Default is 80.
            learning_rate (float): Learning rate for optimizer. Default is 0.0005.
            weight_decay (float): Weight decay for regularization. Default is 3e-4.
            patience (int): Early stopping patience. Default is 20.
            label_smoothing (float): Label smoothing factor. Default is 0.05.
            mixup_alpha (float): Mixup alpha parameter. Default is 0.2.

        Returns:
            tuple: (history, model_path)
                - history: Dictionary containing training history
                - model_path: Path to the saved best model
        """
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
        # This often leads to better generalization and faster convergence
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

                # Apply mixup augmentation
                if mixup_alpha > 0:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    index = torch.randperm(inputs.size(0)).to(self.device)
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                    labels_a, labels_b = labels, labels[index]

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass with mixed inputs
                    outputs = self.second_stage_model(mixed_inputs)
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

                    # For accuracy calculation, we'll use the primary labels
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                else:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.second_stage_model(inputs)
                    loss = criterion(outputs, labels)

                    # Track statistics
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.second_stage_model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update learning rate
                scheduler.step()

                # Track statistics
                train_loss += loss.item() * inputs.size(0)

                # Print progress every 5 batches (since batch size is smaller)
                if (batch_idx + 1) % 5 == 0:
                    print(f"Second Stage - Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(self.second_stage_train_loader)} | "
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
    def train(self, epochs_first=30, epochs_second=80, learning_rate_first=0.0003, 
            learning_rate_second=0.0005, weight_decay=3e-4, patience_first=10, 
            patience_second=20, label_smoothing=0.05, mixup_alpha=0.2):
        """Train both stages of the model
        
        Args:
            epochs_first (int): Number of epochs for first stage training. Default is 30.
            epochs_second (int): Number of epochs for second stage training. Default is 80.
            learning_rate_first (float): Learning rate for first stage. Default is 0.0003.
            learning_rate_second (float): Learning rate for second stage. Default is 0.0005.
            weight_decay (float): Weight decay for regularization. Default is 3e-4.
            patience_first (int): Early stopping patience for first stage. Default is 10.
            patience_second (int): Early stopping patience for second stage. Default is 20.
            label_smoothing (float): Label smoothing factor for second stage. Default is 0.05.
            mixup_alpha (float): Mixup alpha parameter for second stage. Default is 0.2.
            
        Returns:
            tuple: (history, model_paths)
                - history: Dictionary containing training history for both stages
                - model_paths: Dictionary containing paths to saved model weights
        """
        print("Training first stage model: Normal vs Not Normal")
        first_stage_history = self.train_first_stage(
            epochs=epochs_first,
            learning_rate=learning_rate_first,
            weight_decay=weight_decay,
            patience=patience_first
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