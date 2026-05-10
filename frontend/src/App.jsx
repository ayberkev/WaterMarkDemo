import { useState } from 'react';

export default function App() {
  const [page, setPage] = useState('home');
  const [imageSrc, setImageSrc] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setFile(file);
    setError('');

    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Lütfen geçerli bir resim dosyası yükleyin.');
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setError("Resim boyutu 5MB'dan küçük olmalıdır.");
      return;
    }

    const reader = new FileReader();
    reader.onload = () => setImageSrc(reader.result);
    reader.readAsDataURL(file);
  };

  const handleRemoveWatermark = async () => {
    if (!imageSrc) return;

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('http://127.0.0.1:5000/process', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.result) {
        setImageSrc(data.result);
      }
    } catch (err) {
      setError('İşlem başarısız.');
    }

    setLoading(false);
  };

  return (
    <div style={styles.app}>
      <nav style={styles.nav}>
        <button style={styles.navButton} onClick={() => setPage('home')}>
          Ana Sayfa
        </button>
        <button style={styles.navButton} onClick={() => setPage('about')}>
          Hakkında
        </button>
      </nav>

      {page === 'home' && (
        <div style={styles.card}>
          <h1 style={styles.title}>Görsel Watermark Temizleyici</h1>

          {/* 2️⃣ Desteklenen Formatlar */}
          <div style={styles.infoBox}>
            <strong>Desteklenen Formatlar</strong>
            <ul style={styles.list}>
              <li>JPG</li>
              <li>PNG</li>
              <li>WEBP</li>
              <li>Maksimum boyut: 5MB</li>
            </ul>
          </div>

          <input type="file" accept="image/*" onChange={handleImageUpload} />

          {error && <p style={styles.error}>{error}</p>}

          {/* 1️⃣ Before / After (Demo) */}
          {imageSrc && (
            <div style={styles.beforeAfter}>
              <div>
                <p style={styles.label}>Önce</p>
                <img src={imageSrc} style={styles.image} />
              </div>
              <div>
                <p style={styles.label}>Sonra (Demo)</p>
                <img
                  src={imageSrc}
                  style={{ ...styles.image, filter: 'blur(4px)' }}
                />
              </div>
            </div>
          )}

          <button
            onClick={handleRemoveWatermark}
            disabled={!imageSrc || loading}
            style={{
              ...styles.actionButton,
              opacity: !imageSrc || loading ? 0.5 : 1,
            }}
          >
            {loading ? 'İşleniyor...' : 'Watermark Kaldır'}
          </button>

          {/* 3️⃣ Etik Kullanım */}
          <p style={styles.ethic}>
            ⚠️ Bu uygulama yalnızca kişisel ve izinli görseller için
            kullanılmalıdır.
          </p>

          {/* 4️⃣ Nasıl Çalışır */}
          <div style={styles.howItWorks}>
            <h3>Nasıl Çalışır?</h3>
            <ol>
              <li>Görsel kullanıcı tarafından yüklenir</li>
              <li>OCR ile watermark alanı tespit edilir</li>
              <li>Tespit edilen alan maskelenir</li>
              <li>Görsel yeniden oluşturulur</li>
            </ol>
            <p style={styles.note}>
              Bu süreç frontend demo olarak simüle edilmektedir.
            </p>
          </div>
        </div>
      )}

      {page === 'about' && (
        <div style={styles.card}>
          <h1 style={styles.title}>Hakkında</h1>
          <p style={styles.aboutText}>
            Ayberk Psav'ın OCR yardımı ile watermark temizleme projesidir.
            <br />
            Bana ayberkpsav@gmail.com üzerinden oluşabilirsiniz.
          </p>
        </div>
      )}
    </div>
  );
}

const styles = {
  app: {
    minHeight: '100vh',
    backgroundColor: '#f4f6f8',
    fontFamily: 'Arial, sans-serif',
    color: '#111',
  },
  nav: {
    display: 'flex',
    justifyContent: 'center',
    gap: '12px',
    padding: '12px',
    backgroundColor: '#111',
  },
  navButton: {
    backgroundColor: '#fff',
    color: '#111',
    border: '1px solid #ccc',
    padding: '10px 18px',
    cursor: 'pointer',
    borderRadius: '6px',
  },
  card: {
    maxWidth: '520px',
    width: '92%',
    margin: '30px auto',
    backgroundColor: '#fff',
    padding: '20px',
    borderRadius: '10px',
    boxShadow: '0 6px 16px rgba(0,0,0,0.15)',
    textAlign: 'center',
  },
  title: {
    marginBottom: '16px',
    fontSize: '22px',
  },
  infoBox: {
    backgroundColor: '#eef2f5',
    padding: '10px',
    marginBottom: '12px',
    borderRadius: '6px',
    fontSize: '14px',
  },
  list: {
    paddingLeft: '18px',
    textAlign: 'left',
  },
  beforeAfter: {
    display: 'flex',
    gap: '10px',
    justifyContent: 'center',
    marginTop: '15px',
    flexWrap: 'wrap',
  },
  label: {
    fontSize: '13px',
    marginBottom: '4px',
  },
  image: {
    maxWidth: '220px',
    borderRadius: '6px',
  },
  actionButton: {
    marginTop: '16px',
    padding: '12px',
    width: '100%',
    cursor: 'pointer',
    backgroundColor: '#111',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
  },
  error: {
    color: '#c40000',
    marginTop: '10px',
  },
  ethic: {
    fontSize: '13px',
    marginTop: '12px',
    color: '#444',
  },
  howItWorks: {
    textAlign: 'left',
    marginTop: '16px',
    fontSize: '14px',
  },
  note: {
    fontSize: '12px',
    color: '#555',
  },
  aboutText: {
    fontSize: '15px',
    lineHeight: '1.6',
  },
};
