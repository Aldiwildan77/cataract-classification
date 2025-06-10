import numpy as np
import cv2

def get_average_color_channels(img):
    if img is None:
        raise ValueError("Image not found or could not be read.")
    red_channel = img[:, :, 2]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 0]

    red_sum = np.sum(red_channel)
    green_sum = np.sum(green_channel)
    blue_sum = np.sum(blue_channel)

    red_avg = red_sum / red_channel.size
    green_avg = green_sum / green_channel.size
    blue_avg = blue_sum / blue_channel.size

    return red_avg, green_avg, blue_avg

def get_brightness(img):
    if img is None:
        raise ValueError("Image not found or could not be read.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness

def get_contrast(img):
    if img is None:
        raise ValueError("Image not found or could not be read.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    return contrast

def get_sharpness(img):
    if img is None:
        raise ValueError("Image not found or could not be read.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness

def get_total_channel_average(dataset_pd):
    total_r = dataset_pd['average_r'].sum()
    total_g = dataset_pd['average_g'].sum()
    total_b = dataset_pd['average_b'].sum()

    total_images = len(dataset_pd)

    average_r = total_r / total_images
    average_g = total_g / total_images
    average_b = total_b / total_images

    return average_r, average_g, average_b

def get_total_channel_std(dataset_pd):
    std_r = dataset_pd['average_r'].std()
    std_g = dataset_pd['average_g'].std()
    std_b = dataset_pd['average_b'].std()

    return std_r, std_g, std_b

def calculate_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mean_value = series.mean()
    return [iqr, lower_bound, upper_bound, mean_value]

def get_outliers_iqr(data, column_name):
    col_data = data[column_name]
    iqr, lower_bound, upper_bound, mean_value = calculate_iqr(col_data)
    return data[(col_data < lower_bound) | (col_data > upper_bound)]
