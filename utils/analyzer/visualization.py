from matplotlib import pyplot as plt
import seaborn as sns
import math

import cv2
import os
import pandas as pd

import numpy as np

from utils.analyzer.image import calculate_iqr, get_outliers_iqr


def draw_box_plot_iqr(dataset_pd, column_names):
    max_cols = 6
    n_cols = min(max_cols, len(column_names))
    n_rows = math.ceil(len(column_names) / max_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 6 * n_rows), squeeze=False)

    for idx, column_name in enumerate(column_names):
        row, col = divmod(idx, max_cols)
        ax = axes[row][col]

        iqr, lower_bound, upper_bound, mean_value = calculate_iqr(dataset_pd[column_name])
        col_data = dataset_pd[column_name]

        sns.boxplot(y=col_data, ax=ax, color='lightblue', fliersize=3, medianprops=dict(color='orange'))

        ax.axhline(lower_bound, color='red', linestyle='--', linewidth=1, label='Lower Bound')
        ax.axhline(upper_bound, color='green', linestyle='--', linewidth=1, label='Upper Bound')
        ax.axhline(mean_value, color='purple', linestyle='--', linewidth=1, label='Mean')

        outliers = get_outliers_iqr(dataset_pd, column_name)
        ax.scatter(
            x=np.zeros(len(outliers)),
            y=outliers[column_name].values,
            color='red', label='Outliers', s=30, edgecolor='black'
        )

        # Always print total outliers
        total_outliers = len(outliers)
        ax.set_title(f'{column_name} distribution\nTotal outliers: {total_outliers}', fontsize=11)

        ax.set_xlabel('')
        ax.set_ylabel(column_name, fontsize=10)
        ax.grid(axis='y', linestyle=':', linewidth=0.5)
        ax.set_xticks([])

    # Hide unused subplots
    for idx in range(len(column_names), n_rows * n_cols):
        fig.delaxes(axes[idx // max_cols][idx % max_cols])

    # Show shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def draw_histogram_distribution(dataset_pd, column_names):
    max_cols = 3
    n_cols = min(max_cols, len(column_names))
    n_rows = math.ceil(len(column_names) / max_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, column_name in enumerate(column_names):
        row, col = divmod(idx, max_cols)
        ax = axes[row][col]

        sns.histplot(data=dataset_pd, x=column_name, kde=True, bins=30, color='skyblue', ax=ax)

        ax.set_title(f'{column_name} Histogram', fontsize=12)
        ax.set_xlabel(column_name)
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', linestyle=':', linewidth=0.5)

    # Hide unused subplots
    for idx in range(len(column_names), n_rows * n_cols):
        fig.delaxes(axes[idx // max_cols][idx % max_cols])

    plt.tight_layout()
    plt.show()

def analyze_image_quality_batch_with_flags(folder_path, limit=None, blurry_thresh=10, dark_thresh=50, overexposed_thresh=220):
    """
    Analyze image quality metrics for all images in a folder and flag bad images.

    Parameters:
        folder_path (str): Path to the folder containing images.
        limit (int, optional): Limit the number of images to process.
        blurry_thresh (float): Sharpness threshold below which images are considered blurry.
        dark_thresh (float): Brightness threshold below which images are considered too dark.
        overexposed_thresh (float): Brightness threshold above which images are considered overexposed.

    Returns:
        pd.DataFrame: DataFrame containing image quality metrics with flags.
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if limit:
        image_files = image_files[:limit]

    data = []

    for filename in image_files:
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        exposure = brightness / 255.0

        flags = {
            "is_blurry": sharpness < blurry_thresh,
            "is_dark": brightness < dark_thresh,
            "is_overexposed": brightness > overexposed_thresh
        }

        data.append({
            "filename": str(filename),
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "exposure": exposure,
            **flags
        })

    df = pd.DataFrame(data)

    # Show histogram summary
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Batch Image Quality Metrics Histogram with Flags', fontsize=16)

    sns.histplot(df['brightness'], bins=30, ax=axes[0][0], color='gray')
    axes[0][0].set_title('Brightness')

    sns.histplot(df['contrast'], bins=30, ax=axes[0][1], color='orange')
    axes[0][1].set_title('Contrast')

    sns.histplot(df['sharpness'], bins=30, ax=axes[1][0], color='purple')
    axes[1][0].set_title('Sharpness')

    sns.histplot(df['exposure'], bins=30, ax=axes[1][1], color='cyan')
    axes[1][1].set_title('Exposure')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return df

def draw_image_histogram_metrics(image_input):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load image if path is given
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_input.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    exposure = np.mean(gray) / 255.0

    # Luminance histogram (DSLR style)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Setup plots
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle("Image Metric Histograms (RGB + Brightness + Contrast + Sharpness + Exposure + Luminance)", fontsize=16)

    # RGB
    color_labels = ['Red', 'Green', 'Blue']
    for i, color in enumerate(color_labels):
        channel = image[:, :, i]
        ax = axes[0][i]
        sns.histplot(channel.ravel(), bins=256, color=color.lower(), ax=ax)
        ax.set_title(f'{color} Channel')
        ax.set_xlim([0, 255])

    # Brightness
    sns.histplot(gray.ravel(), bins=256, color='gray', ax=axes[1][0])
    axes[1][0].set_title(f'Brightness\nMean: {brightness:.2f}')
    axes[1][0].set_xlim([0, 255])

    # Contrast
    sns.histplot((gray - brightness).ravel(), bins=256, color='orange', ax=axes[1][1])
    axes[1][1].set_title(f'Contrast\nStd Dev: {contrast:.2f}')

    # Sharpness & Exposure bar chart
    axes[1][2].bar(['Sharpness', 'Exposure'], [sharpness, exposure], color=['purple', 'cyan'])
    axes[1][2].set_ylim(0, max(sharpness, exposure, 1.0) * 1.2)
    axes[1][2].set_title(f'Sharpness: {sharpness:.2f} | Exposure: {exposure:.2f}')

    # DSLR-style Luminance Histogram
    sns.histplot(luminance.ravel(), bins=256, color='black', ax=axes[2][0])
    axes[2][0].set_title('Luminance (DSLR Style)')
    axes[2][0].set_xlim([0, 255])

    # Hide unused plots
    axes[2][1].axis('off')
    axes[2][2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
