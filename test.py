import torch
import torchvision.models as models
import torchvision.transforms as T
# torchvision.datasets.QuickDraw is not used anymore
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time
import struct # For unpacking binary data
from struct import unpack
import os
# import urllib.request # Removed: No longer downloading
from PIL import Image, ImageDraw
# from tqdm import tqdm # Import tqdm

# --- QuickDraw Binary Data Reading Functions (from user - Unchanged) ---
def unpack_drawing(file_handle):
    try:
        key_id, = unpack('Q', file_handle.read(8))
        country_code, = unpack('2s', file_handle.read(2))
        recognized, = unpack('b', file_handle.read(1))
        timestamp, = unpack('I', file_handle.read(4))
        n_strokes, = unpack('H', file_handle.read(2))
        image_strokes = []
        for _ in range(n_strokes):
            n_points, = unpack('H', file_handle.read(2))
            fmt = str(n_points) + 'B'
            if n_points == 0:
                image_strokes.append(((), ()))
                continue
            x = unpack(fmt, file_handle.read(n_points))
            y = unpack(fmt, file_handle.read(n_points))
            image_strokes.append((x, y))

        return {
            'key_id': key_id,
            'country_code': country_code,
            'recognized': recognized,
            'timestamp': timestamp,
            'image': image_strokes
        }
    except struct.error as e:
        raise struct.error
    except Exception as e:
        raise


def unpack_drawings(filename):
    # Get file size for tqdm progress bar
    file_size = os.path.getsize(filename)
    with open(filename, 'rb') as f, tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Unpacking {os.path.basename(filename)}", leave=False) as pbar:
        while True:
            try:
                start_pos = f.tell()
                yield unpack_drawing(f)
                pbar.update(f.tell() - start_pos)
            except struct.error:
                break
            except EOFError:
                break

# --- Custom QuickDraw Dataset from Local Binary Files (MODIFIED) ---
class QuickDrawBinaryDataset(Dataset):
    IMAGE_SIZE = (256, 256)
    LINE_WIDTH = 2

    def __init__(self, root, category, transform=None): # Removed download parameter
        self.root = root # This should be the path to the directory containing .bin files, e.g., './data'
        self.category = category.replace(' ', '_') # Ensure category name is filename-safe
        self.transform = transform
        # Construct filepath assuming .bin files are directly in the root directory
        self.filepath = os.path.join(self.root, f"full_binary_{self.category}.bin")

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Dataset binary file not found: {self.filepath}. Please ensure it exists in the specified directory."
            )
        
        logger.info(f"Loading drawings from {self.filepath} for category {self.category}...")
        self.drawings = list(unpack_drawings(self.filepath))
        if not self.drawings:
            logger.warning(f"No drawings loaded for category {self.category} from {self.filepath}.")

    # Removed download_file method

    def _render_drawing_to_image(self, drawing_strokes):
        image = Image.new("L", self.IMAGE_SIZE, "white")
        draw = ImageDraw.Draw(image)
        for stroke_x, stroke_y in drawing_strokes:
            if len(stroke_x) < 1:
                continue
            if len(stroke_x) == 1:
                draw.point((stroke_x[0], stroke_y[0]), fill="black")
            else:
                points = list(zip(stroke_x, stroke_y))
                draw.line(points, fill="black", width=self.LINE_WIDTH)
        return image

    def __len__(self):
        return len(self.drawings)

    def __getitem__(self, idx):
        drawing_data = self.drawings[idx]
        pil_image = self._render_drawing_to_image(drawing_data['image'])

        if self.transform:
            pil_image = self.transform(pil_image)
        
        return pil_image, self.category 

# Configuration
QUICKDRAW_CATEGORIES = ['apple', 'banana', 'cat', 'dog', 'car']
NUM_TRAIN_SAMPLES_PER_CATEGORY = 5000
NUM_TEST_SAMPLES_PER_CATEGORY = 1000
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BINARY_DATA_ROOT = './data' # MODIFIED: Point to local data directory

# --- Model Definitions and Feature Extractors (Unchanged) ---
MODELS_TO_TEST = {
    "MobileNetV3-Small": {
        "weights": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "model_fn": models.mobilenet_v3_small,
        "feature_extractor_fn": lambda m: MobileNetV3FeatureExtractor(m)
    },
    "ShuffleNetV2_x0_5": {
        "weights": models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1,
        "model_fn": models.shufflenet_v2_x0_5,
        "feature_extractor_fn": lambda m: ShuffleNetV2FeatureExtractor(m)
    },
    "SqueezeNet1_1": {
        "weights": models.SqueezeNet1_1_Weights.IMAGENET1K_V1,
        "model_fn": models.squeezenet1_1,
        "feature_extractor_fn": lambda m: SqueezeNetFeatureExtractor(m)
    },
    "EfficientNet-B0": {
        "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "model_fn": models.efficientnet_b0,
        "feature_extractor_fn": lambda m: EfficientNetFeatureExtractor(m)
    }
}

class MobileNetV3FeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.flatten = torch.nn.Flatten(1)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

class ShuffleNetV2FeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.conv1 = original_model.conv1
        self.maxpool = original_model.maxpool
        self.stage2 = original_model.stage2
        self.stage3 = original_model.stage3
        self.stage4 = original_model.stage4
        self.conv5 = original_model.conv5
        self.glob_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.flatten = torch.nn.Flatten(1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        x = self.flatten(x)
        return x

class SqueezeNetFeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten(1)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

class EfficientNetFeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.flatten = torch.nn.Flatten(1)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

# --- Data Loading and Preprocessing (MODIFIED) ---
def get_quickdraw_data(categories, num_train_per_cat, num_test_per_cat, base_model_transform, data_root):
    all_train_datasets = []
    all_test_datasets = []

    quickdraw_specific_transform = T.Compose([
        T.Grayscale(num_output_channels=3),
        base_model_transform
    ])

    for i, category_name in enumerate(tqdm(categories, desc="Processing categories", unit="category")):
        logger.info(f"Attempting to load QuickDraw category: {category_name} from local binary files...")
        try:
            full_category_dataset = QuickDrawBinaryDataset(
                root=data_root, # This will be './data'
                category=category_name,
                transform=quickdraw_specific_transform
                # download=True argument removed
            )

            if len(full_category_dataset) == 0:
                logger.warning(f"No data loaded for category {category_name}. Skipping.")
                continue
            
            actual_num_train = min(num_train_per_cat, len(full_category_dataset))
            remaining_samples = len(full_category_dataset) - actual_num_train
            actual_num_test = min(num_test_per_cat, remaining_samples)

            if actual_num_train + actual_num_test > len(full_category_dataset):
                 logger.warning(f"Not enough samples in {category_name} for desired train/test split. Adjusting.")
                 actual_num_train = min(num_train_per_cat, len(full_category_dataset))
                 actual_num_test = min(num_test_per_cat, len(full_category_dataset) - actual_num_train)
                 if actual_num_test < 0: actual_num_test = 0

            if actual_num_train == 0 and num_train_per_cat > 0:
                logger.warning(f"Could not get any training samples for {category_name}")
            if actual_num_test == 0 and num_test_per_cat > 0:
                 logger.warning(f"Could not get any testing samples for {category_name} after allocating for training.")

            indices = np.arange(len(full_category_dataset))
            np.random.shuffle(indices)
            
            train_indices = indices[:actual_num_train]
            test_indices = indices[actual_num_train : actual_num_train + actual_num_test]

            class LabeledSubset(Subset):
                def __init__(self, dataset, indices, label):
                    super().__init__(dataset, indices)
                    self.label = label
                def __getitem__(self, idx):
                    data, _ = super().__getitem__(idx)
                    return data, self.label

            if len(train_indices) > 0:
                all_train_datasets.append(LabeledSubset(full_category_dataset, train_indices, i))
            if len(test_indices) > 0:
                all_test_datasets.append(LabeledSubset(full_category_dataset, test_indices, i))

        except (FileNotFoundError, RuntimeError) as e: # Catch load errors
            logger.error(f"Could not load category {category_name}. Error: {e}. Skipping.")
            continue
    
    if not all_train_datasets or not all_test_datasets:
        if not any(all_train_datasets) and not any(all_test_datasets):
             logger.critical("No QuickDraw data could be loaded for any category. Aborting.")
             raise RuntimeError("No QuickDraw data could be loaded for any category. Aborting.")
        else:
            logger.warning("Some categories failed to load, proceeding with available data.")

    all_train_datasets = [ds for ds in all_train_datasets if ds is not None and len(ds) > 0]
    all_test_datasets = [ds for ds in all_test_datasets if ds is not None and len(ds) > 0]

    if not all_train_datasets or not all_test_datasets:
        logger.critical("No usable QuickDraw train/test data after processing categories. Aborting.")
        raise RuntimeError("No usable QuickDraw train/test data after processing categories. Aborting.")

    return ConcatDataset(all_train_datasets), ConcatDataset(all_test_datasets)


# --- Feature Extraction (MODIFIED) ---
def extract_features(model, dataloader, device, description="Extracting features"):
    model.eval()
    model.to(device)
    features_list = []
    labels_list = []
    for inputs, labels in tqdm(dataloader, desc=description, leave=False, unit="batch"):
        inputs = inputs.to(device)
        with torch.no_grad(): # Ensure no gradients are computed in this block
            outputs = model(inputs)
        features_list.append(outputs.cpu().detach().numpy()) # Detach before converting to numpy
        # Labels are typically not part of the computation graph, but detaching is harmless
        if isinstance(labels, torch.Tensor):
            labels_list.append(labels.cpu().detach().numpy())
        else: # If labels are already numpy arrays or other types
            labels_list.append(labels)
    
    if not features_list:
        return np.array([]), np.array([])

    return np.concatenate(features_list), np.concatenate(labels_list)

# --- Main Benchmarking Loop (MODIFIED calls to extract_features) ---
def run_benchmark():
    results = []
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Benchmarking on QuickDraw categories: {', '.join(QUICKDRAW_CATEGORIES)}")
    logger.info(f"Samples per category: {NUM_TRAIN_SAMPLES_PER_CATEGORY} train, {NUM_TEST_SAMPLES_PER_CATEGORY} test.")
    logger.info(f"Loading QuickDraw data from local directory: {os.path.abspath(BINARY_DATA_ROOT)}\n")


    for model_name, config in tqdm(MODELS_TO_TEST.items(), desc="Benchmarking Models", unit="model"):
        logger.info(f"--- Testing Model: {model_name} ---")
        
        current_model_accuracy = "Error"
        current_model_acc_per_param = "Error"
        current_model_feat_ext_time = "N/A"
        current_model_train_time = "N/A"
        model_params = "Error" # Initialize in case model loading fails

        try:
            weights = config["weights"]
            base_model = config["model_fn"](weights=weights)
            model_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad) / 1_000_000

            model_specific_transforms = weights.transforms()
            
            train_dataset, test_dataset = get_quickdraw_data(
                QUICKDRAW_CATEGORIES,
                NUM_TRAIN_SAMPLES_PER_CATEGORY,
                NUM_TEST_SAMPLES_PER_CATEGORY,
                model_specific_transforms,
                BINARY_DATA_ROOT
            )
            
            if len(train_dataset) == 0 or len(test_dataset) == 0:
                logger.warning(f"Not enough data loaded for {model_name} to proceed. Skipping.")
                raise RuntimeError("Empty dataset after loading.")

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            feature_extractor = config["feature_extractor_fn"](base_model)
            feature_extractor.to(DEVICE)
            feature_extractor.eval()

            logger.info("Extracting training features...")
            start_time = time.time()
            train_features, train_labels = extract_features(feature_extractor, train_loader, DEVICE, description=f"Extracting train features ({model_name})")
            feat_ext_time_train = time.time() - start_time
            
            if train_features.size == 0:
                logger.warning(f"No training features extracted for {model_name}. Skipping further steps for this model.")
                raise RuntimeError("No training features.")
            logger.debug(f"Train features shape: {train_features.shape}")

            logger.info("Extracting test features...")
            start_time = time.time()
            test_features, test_labels = extract_features(feature_extractor, test_loader, DEVICE, description=f"Extracting test features ({model_name})")
            feat_ext_time_test = time.time() - start_time

            if test_features.size == 0:
                logger.warning(f"No test features extracted for {model_name}. Skipping further steps for this model.")
                raise RuntimeError("No test features.")
            logger.debug(f"Test features shape: {test_features.shape}")
            
            current_model_feat_ext_time = f"{feat_ext_time_train + feat_ext_time_test:.2f}"

            logger.info("Training Logistic Regression classifier...")
            start_time = time.time()
            classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42, solver='liblinear', C=0.1))
            classifier.fit(train_features, train_labels)
            current_model_train_time = f"{time.time() - start_time:.2f}"

            accuracy = classifier.score(test_features, test_labels) * 100
            logger.info(f"QuickDraw Test Accuracy for {model_name}: {accuracy:.2f}%")
            current_model_accuracy = f"{accuracy:.2f}"
            current_model_acc_per_param = f"{accuracy / model_params if isinstance(model_params, (int, float)) and model_params > 0 else 0:.2f}"


        except RuntimeError as e:
            logger.error(f"Runtime error during processing for {model_name}: {e}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred for model {model_name}: {e}") # Use logger.exception for traceback
        finally:
            results.append({
                "Model": model_name,
                "Accuracy (%)": current_model_accuracy,
                "Params (M)": f"{model_params:.2f}" if isinstance(model_params, (int, float)) else model_params,
                "Acc/Params": current_model_acc_per_param,
                "Feat Ext Time (s)": current_model_feat_ext_time,
                "Train Time (s)": current_model_train_time
            })
            logger.info("-" * 30 + "\n")
            if 'base_model' in locals(): del base_model
            if 'feature_extractor' in locals(): del feature_extractor
            if 'train_dataset' in locals(): del train_dataset
            if 'test_dataset' in locals(): del test_dataset
            if 'train_loader' in locals(): del train_loader
            if 'test_loader' in locals(): del test_loader
            if 'train_features' in locals(): del train_features
            if 'train_labels' in locals(): del train_labels
            if 'test_features' in locals(): del test_features
            if 'test_labels' in locals(): del test_labels
            if DEVICE == torch.device("cuda"):
                torch.cuda.empty_cache()

    print("\n--- Benchmark Results ---") # Keep this as print for final console summary
    if not results:
        print("No results to display.") # Keep as print
        return

    # --- Display results in console ---
    headers = results[0].keys()
    col_widths = {key: len(key) for key in headers}
    for row in results:
        for key in headers:
            col_widths[key] = max(col_widths.get(key, 0), len(str(row.get(key, ""))))
    
    for key in col_widths:
        col_widths[key] += 2

    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in results:
        row_line = " | ".join(f"{str(row.get(h, 'N/A')):<{col_widths[h]}}" for h in headers)
        print(row_line)

    # --- Save results to CSV ---
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"benchmark_results_{timestamp}.csv"
        try:
            with open(results_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(results)
            logger.info(f"Benchmark results saved to {results_filename}")
        except IOError as e:
            logger.error(f"Could not save results to CSV: {e}")


if __name__ == '__main__':
    # Logging is configured at the top of the script
    run_benchmark()