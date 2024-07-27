import os
import cv2
import pickle
from skimage.metrics import structural_similarity as ssim

# jika dua gambar memiliki korelasi di atas 0.5 akan dianggap cukup mirip
correl_threshold = 0.5

# jika dua gambar memiliki indeks kesamaan di atas 0.0 akan dianggap cukup mirip
# nilai ini sangat rendah agar memastikan bahwa semua kesamaan dicatat
similarity_index_threshold = 0.0

# mengatur batas maksimum jumlah pasangan kesamaan menggunakan SSIM
ssim_matches_limit = 100

# mengatur batas maksimum jumlah fitur SIFT yang akan diekstraksi dari gambar
sift_features_limit = 1000

# mengatur rasio Lowe untuk pencocokan fitur SIFT
# dianggap sebagai pencocokan yang baik jika rasio jarak dari pencocokan terbaik ke pencocokan kedua terbaik adalah di bawah 0.75
# ini membantu mengurangi pencocokan yang salah.
lowe_ratio = 0.75

# mengatur jumlah prediksi yang akan dihasilkan
predictions_count = 4


def get_train_images():
    # menginisalisasi sebuah daftar kosong yang akan digunakan untuk menyimpan path gambar
    train_paths = []

    # mendefinisikan variabel train_path dengan nilai train_images
    train_path = "train_images/"

    # memulai loop untuk menelusuri direktori train_path secara rekursif
    for root, dirs, files in os.walk(train_path):

        # memulai loop untuk iterasi melalui setiap file dalam daftar files
        for file in files:
            # menggabungkan root dan file menjadi satu path lengkap menggunakan os.path.join
            # lalu menambahkan path tersebut ke daftar train_paths
            train_paths.append((os.path.join(root, file)))

    # mengembalikan daftar train_paths yang berisi path lengkap semua file gambar
    return train_paths


# Mendefinisikan sebuah fungsi bernama yang menerima satu argumen hist_train
def write_train_images_histogram_to_pickle(hist_train):
    # membuka file untuk ditulis dalam mode biner, yang diperlukan untuk menyimpan objek dengan pickle
    # memberikan alias f untuk file objek sehingga dapat digunakan di dalam blok with
    with open('train_hist_data.pkl', 'wb') as f:
        # menggunakan modul pickle untuk menyimpan objek hist_train ke file yang telah dibuka
        pickle.dump(hist_train, f)


def calculate_train_images_histogram():
    # menginisialisasi sebuah daftar kosong
    hist_train = []

    # memanggil fungsi get_train_images() untuk mendapatkan daftar path gambar
    train_paths = get_train_images()

    # memulai loop untuk iterasi melalui setiap path dalam train_paths
    for path in train_paths:

        # membaca gambar dari path saat ini
        image = cv2.imread(path)

        #  jika image bernilai None
        #  lewati iterasi ini dan lanjutkan ke path berikutnya
        if image is None:
            continue

        # mengonversi gambar dari format BGR ke format RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #  menghitung histogram gambar dalam ruang warna RGB.
        # [image]: gambar input untuk menghitung histogram.
        # [0, 1, 2]: saluran warna yang dihitung (R, G, B).
        # None: Mask tidak digunakan.
        # [8, 8, 8]: jumlah bin untuk setiap saluran warna.
        # [0, 256, 0, 256, 0, 256]: rentang nilai intensitas untuk setiap saluran.
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # menormalkan histogram sehingga nilai-nilai dalam histogram berada dalam rentang yang konsisten
        hist = cv2.normalize(hist, None)

        # menambahkan pasangan path dan histogram ke dalam daftar hist_train
        hist_train.append((path, hist))

    # memanggil fungsi write_train_images_histogram_to_pickle
    # dengan hist_train sebagai argumen untuk menyimpan data histogram ke file pickle
    write_train_images_histogram_to_pickle(hist_train)


def load_train_images_histogram_from_pickle():
    # membuka file bernama train_hist_data.pkl dalam mode baca biner
    with open('train_hist_data.pkl', 'rb') as f:
        # memuat data dari file pickle
        hist_train = pickle.load(f)

        # mengembalikan variabel hist_train yang berisi data histogram gambar
        return hist_train


# mendefinisikan fungsi yang menerima argumen image_name
def calculate_query_image_histogram(image_name):
    # menyusun path lengkap ke gambar kueri dengan menggabungkan folder query_images dan nama gambar
    path = f"query_images/{image_name}"

    # membaca gambar dari path yang diberikan
    image = cv2.imread(path)

    # mengonversi gambar dari format BGR ke format RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # membuat objek CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # memisahkan gambar menjadi tiga saluran warna (R, G, B).
    channels = cv2.split(image)

    # menerapkan CLAHE pada setiap saluran warna (R, G, B).
    clahe_channels = [clahe.apply(channel) for channel in channels]

    # menggabungkan kembali saluran yang telah diolah dengan CLAHE menjadi satu gambar
    image = cv2.merge(clahe_channels)

    #  menghitung histogram gambar dalam ruang warna RGB.
    # [image]: gambar input untuk menghitung histogram.
    # [0, 1, 2]: saluran warna yang dihitung (R, G, B).
    # None: Mask tidak digunakan.
    # [8, 8, 8]: jumlah bin untuk setiap saluran warna.
    # [0, 256, 0, 256, 0, 256]: rentang nilai intensitas untuk setiap saluran.
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # menormalkan histogram sehingga nilai-nilai dalam histogram berada dalam rentang yang konsisten
    hist = cv2.normalize(hist, None)

    # menambahkan pasangan path dan histogram ke dalam daftar hist_query
    hist_query = (path, hist)

    # mengembalikan gambar histogram query image
    return hist_query


#  mendefinisikan fungsi find_matching_histogram yang menerima satu argumen image_name
def find_matching_histogram(image_name):
    # menghitung histogram untuk gambar query menggunakan fungsi calculate_query_image_histogram
    hist_query = calculate_query_image_histogram(image_name)

    # memuat histogram gambar dari file pickle menggunakan fungsi load_train_images_histogram_from_pickle
    hist_train = load_train_images_histogram_from_pickle()

    # menginisialisasi daftar kosong matches untuk menyimpan pasangan histogram yang cocok.
    matches = []

    # memulai loop untuk iterasi melalui setiap histogram dalam hist_train
    for j in range(len(hist_train)):

        # membandingkan histogram kueri dengan histogram pelatihan menggunakan metode korelasi cv2.compareHist
        cmp = cv2.compareHist(hist_query[1], hist_train[j][1], cv2.HISTCMP_CORREL)

        # mengecek apakah nilai cmp lebih besar dari ambang batas korelasi
        if cmp > correl_threshold:

            # menambahkan tuple ke dalam daftar matches jika histogram cukup mirip
            matches.append((cmp, hist_train[j][0]))

    # mengurutkan daftar matches berdasarkan nilai korelasi dalam urutan menurun
    matches.sort(key=lambda x: x[0], reverse=True)

    # membuat tuple hist_matches yang berisi path gambar kueri dan daftar pasangan histogram yang cocok
    hist_matches = (hist_query[0], matches)

    # mengembalikan tuple hist_matches yang berisi path gambar kueri dan daftar pasangan histogram yang cocok
    return hist_matches


# mendefinisikan fungsi ssim_index yang menerima dua argumen
# yaitu path gambar kueri dan path gambar pelatihan
def ssim_index(q_path, m_path):

    # membaca gambar dari path q_path
    q_i = cv2.imread(q_path, 0)

    # mengubah ukuran gambar kueri menjadi 8x8 piksel
    q_i = cv2.resize(q_i, (8, 8))

    # membaca gambar dari path m_path
    m_i = cv2.imread(m_path, 0)

    # mengubah ukuran gambar kueri menjadi 8x8 piksel
    m_i = cv2.resize(m_i, (8, 8))

    # menghitung nilai SSIM antara gambar kueri dan gambar pelatihan
    return ssim(q_i, m_i)


# mendefinisikan fungsi find_ssim_matches yang menerima argumen image_name.
def find_ssim_matches(image_name):
    # menginisialisasi daftar kosong ssim_matches untuk menyimpan hasil pencocokan SSIM
    ssim_matches = []

    # memanggil fungsi find_matching_histogram untuk mendapatkan pasangan histogram yang cocok dengan gambar kueri
    hist_matches = find_matching_histogram(image_name)

    # menyimpan path gambar kueri dari hasil pencocokan histogram
    query_image_path = hist_matches[0]

    # menginisialisasi daftar kosong matches untuk menyimpan pasangan gambar yang cocok berdasarkan SSIM sementara
    matches = []

    # memulai loop untuk iterasi melalui setiap pasangan histogram yang cocok.
    for j in range(len(hist_matches[1])):

        # menyimpan path gambar pelatihan dari hasil pencocokan histogram yang sedang diproses
        match_image_path = hist_matches[1][j][1]

        # menghitung SSIM antara gambar kueri dan gambar pelatihan menggunakan fungsi ssim_index
        si = ssim_index(query_image_path, match_image_path)

        # mengecek apakah nilai SSIM lebih besar dari ambang batas kesamaan
        if si > similarity_index_threshold:

            # menambahkan tuple ke dalam daftar matches jika gambar cukup mirip
            matches.append((si, match_image_path))

    # mengurutkan daftar matches berdasarkan nilai SSIM dalam urutan menurun
    matches.sort(key=lambda x: x[0], reverse=True)

    # menambahkan tuple ke dalam daftar ssim_matches
    ssim_matches.append((query_image_path, matches[:ssim_matches_limit]))

    # mengembalikan daftar ssim_matches yang berisi path gambar kueri dan daftar pasangan gambar yang cocok berdasarkan SSIM
    return ssim_matches


# mendefinisikan fungsi yang menerima argumen image
def gen_sift_features(image):
    # membuat objek SIFT menggunakan OpenCV
    sift = cv2.SIFT.create(sift_features_limit)

    # mendeteksi dan menghitung fitur SIFT dari gambar yang diberikan
    # hasil gungsi detectAndCompute mengembalikan dua nilai
    # kp (keypoints): daftar titik kunci yang terdeteksi dalam gambar
    # setiap titik kunci mengandung informasi tentang posisi, skala, dan orientasi
    # desc (descriptors): Deskriptor yang sesuai dengan setiap titik kunci
    # deskriptor adalah vektor fitur yang menggambarkan penampilan lokal di sekitar setiap titik kunci
    kp, desc = sift.detectAndCompute(image, None)

    # mengembalikan daftar titik kunci dan deskriptor yang dihasilkan dari gambar
    return kp, desc


#  mendefinisikan fungsi yang menerima satu argumen image_name
def gen_sift_prediction(image_name):
    # Mendefinisikan parameter untuk algoritma pencocokan FLANN
    # algorithm=0 menunjukkan penggunaan algoritma KDTree
    # trees=5 menunjukkan jumlah pohon KDTree yang akan dibangun
    index_params = dict(algorithm=0, trees=5)

    # parameter untuk konfigurasi pencarian FLANN
    # dengan checks=50 menunjukkan jumlah pemeriksaan selama pencarian
    search_params = dict(checks=50)

    # membuat objek pencocokan berbasis FLANN
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # menginisialisasi daftar kosong untuk menyimpan prediksi.
    predictions = []

    # memanggil fungsi find_ssim_matches untuk menemukan gambar yang memiliki kesamaan dengan gambar kueri berdasarkan SSIM
    ssim_matches = find_ssim_matches(image_name)

    # loop melalui daftar gambar yang cocok berdasarkan SSIM, membaca gambar kueri dan mengkonversinya ke RGB
    # lalu menghasilkan fitur SIFT dari gambar tersebut.
    for i in range(len(ssim_matches)):
        #  menginisialisasi daftar kosong untuk menyimpan hasil pencocokan FLANN
        matches_flann = []
        q_path = ssim_matches[i][0]
        q_img = cv2.imread(q_path)
        if q_img is None:
            continue
        q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
        q_kp, q_des = gen_sift_features(q_img)
        if q_des is None:
            continue

        # loop melalui daftar gambar yang cocok berdasarkan SSIM untuk setiap gambar kueri
        # membaca gambar pelatihan dan mengkonversinya ke RGB
        # lalu menghasilkan fitur SIFT dari gambar tersebut dan mencocokkannya menggunakan FLANN.
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
        # mengurutkan hasil pencocokan FLANN berdasarkan jumlah pencocokan dalam urutan menurun
        # menambahkan hasil terbaik ke daftar prediksi.
        matches_flann.sort(key=lambda x: x[0], reverse=True)
        predictions.append((q_path, matches_flann[:predictions_count]))

    # mengembalikan daftar prediksi.
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
