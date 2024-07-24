import os
import cv2
import pickle
from skimage.metrics import structural_similarity as ssim

correl_threshold = 0.5
similarity_index_threshold = 0.0
ssim_matches_limit = 100
sift_features_limit = 1000
lowe_ratio = 0.75
predictions_count = 4

FLANN_INDEX_KDTREE = 0

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


def get_train_images():
    train_paths = []
    train_path = "train_images/"

    for root, dirs, files in os.walk(train_path):
        for file in files:
            train_paths.append((os.path.join(root, file)))

    return train_paths


def write_train_images_histogram_to_pickle(hist_train):
    with open('train_hist_data.pkl', 'wb') as f:
        pickle.dump(hist_train, f)


def calculate_train_images_histogram():
    hist_train = []
    train_paths = get_train_images()

    for path in train_paths:
        image = cv2.imread(path)

        if image is None:
            continue

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels = cv2.split(image)
        clahe_channels = [clahe.apply(channel) for channel in channels]
        image = cv2.merge(clahe_channels)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None)
        hist_train.append((path, hist))

    write_train_images_histogram_to_pickle(hist_train)


def load_train_images_histogram_from_pickle():
    with open('train_hist_data.pkl', 'rb') as f:
        hist_train = pickle.load(f)
        return hist_train


def calculate_query_image_histogram(image_name):
    path = f"query_images/{image_name}"
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None)
    hist_query = (path, hist)

    return hist_query


def find_matching_histogram(image_name):
    hist_query = calculate_query_image_histogram(image_name)
    hist_train = load_train_images_histogram_from_pickle()

    matches = []
    for j in range(len(hist_train)):
        cmp = cv2.compareHist(hist_query[1], hist_train[j][1], cv2.HISTCMP_CORREL)
        if cmp > correl_threshold:
            matches.append((cmp, hist_train[j][0]))
    matches.sort(key=lambda x: x[0], reverse=True)
    hist_matches = (hist_query[0], matches)

    return hist_matches


def ssim_index(q_path, m_path):
    q_i = cv2.imread(q_path, 0)
    q_i = cv2.resize(q_i, (8, 8))
    m_i = cv2.imread(m_path, 0)
    m_i = cv2.resize(m_i, (8, 8))
    return ssim(q_i, m_i)


def find_ssim_matches(image_name):
    ssim_matches = []
    hist_matches = find_matching_histogram(image_name)
    query_image_path = hist_matches[0]

    matches = []
    for j in range(len(hist_matches[1])):
        match_image_path = hist_matches[1][j][1]
        si = ssim_index(query_image_path, match_image_path)
        if si > similarity_index_threshold:
            matches.append((si, match_image_path))
    matches.sort(key=lambda x: x[0], reverse=True)
    ssim_matches.append((query_image_path, matches[:ssim_matches_limit]))

    return ssim_matches


def gen_sift_features(image):
    sift = cv2.SIFT.create(sift_features_limit)
    kp, desc = sift.detectAndCompute(image, None)
    return kp, desc


def gen_sift_prediction(image_name):
    predictions = []
    ssim_matches = find_ssim_matches(image_name)

    for i in range(len(ssim_matches)):
        matches_flann = []
        q_path = ssim_matches[i][0]
        q_img = cv2.imread(q_path)
        if q_img is None:
            continue
        q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
        q_kp, q_des = gen_sift_features(q_img)
        if q_des is None:
            continue

        for j in range(len(ssim_matches[i][1])):
            m_path = ssim_matches[i][1][j][1]
            m_img = cv2.imread(m_path)
            if m_img is None:
                continue
            m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
            m_kp, m_des = gen_sift_features(m_img)
            if m_des is None:
                continue
            matches = flann.knnMatch(q_des, m_des, k=2)
            matches_count = 0
            for x, (m, n) in enumerate(matches):
                if m.distance < lowe_ratio * n.distance:
                    matches_count += 1
            matches_flann.append((matches_count, m_path))
        matches_flann.sort(key=lambda x: x[0], reverse=True)
        predictions.append((q_path, matches_flann[:predictions_count]))

    return predictions


def result(image_name):
    predictions = gen_sift_prediction(image_name)

    if predictions:
        return predictions[0][1]
    else:
        return None


def main(image_name):
    # calculate_train_images_histogram()
    # start_time = time.time()
    return result(image_name)
    # elapsed_time = time.time() - start_time
    # print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == "__main__":
    main()
