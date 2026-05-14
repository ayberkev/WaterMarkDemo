import { useMemo, useRef, useState } from 'react';

const API_URL = 'http://127.0.0.1:5000/process';

function NavButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        ...styles.navButton,
        ...(active ? styles.navButtonActive : {}),
      }}
    >
      {children}
    </button>
  );
}

function SectionTitle({ eyebrow, title, text }) {
  return (
    <div style={styles.sectionHeader}>
      {eyebrow && <span style={styles.eyebrow}>{eyebrow}</span>}
      <h2 style={styles.sectionTitle}>{title}</h2>
      {text && <p style={styles.sectionText}>{text}</p>}
    </div>
  );
}

function PreviewPanel({ label, src, alt, emptyText }) {
  return (
    <div style={styles.previewPanel}>
      <p style={styles.label}>{label}</p>
      {src ? (
        <img src={src} alt={alt} style={styles.image} />
      ) : (
        <div style={styles.emptyPreview}>{emptyText}</div>
      )}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState('home');
  const [previewSrc, setPreviewSrc] = useState(null);
  const [resultSrc, setResultSrc] = useState(null);
  const [maskSrc, setMaskSrc] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [sensitivity, setSensitivity] = useState('medium');
  const fileInputRef = useRef(null);

  const selectedFileLabel = useMemo(() => {
    if (!file) return 'Henüz dosya seçilmedi';
    return `${file.name} • ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  }, [file]);

  const resetState = () => {
    setPreviewSrc(null);
    setResultSrc(null);
    setMaskSrc(null);
    setError('');
    setMessage('');
    setFile(null);
    setLoading(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleImageUpload = (event) => {
    const selectedFile = event.target.files?.[0];

    setFile(null);
    setPreviewSrc(null);
    setResultSrc(null);
    setMaskSrc(null);
    setError('');
    setMessage('');

    if (!selectedFile) return;

    if (!selectedFile.type.startsWith('image/')) {
      setError('Lütfen JPG, PNG veya WEBP gibi geçerli bir görsel dosyası yükleyin.');
      return;
    }

    if (selectedFile.size > 5 * 1024 * 1024) {
      setError("Görsel boyutu 5MB'dan küçük olmalıdır.");
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      setFile(selectedFile);
      setPreviewSrc(reader.result);
    };
    reader.onerror = () => setError('Görsel okunamadı. Lütfen farklı bir dosya deneyin.');
    reader.readAsDataURL(selectedFile);
  };

  const handleRemoveWatermark = async () => {
    if (!file) {
      setError('Önce bir görsel yüklemelisiniz.');
      return;
    }

    setLoading(true);
    setError('');
    setMessage('');
    setResultSrc(null);
    setMaskSrc(null);

    const formData = new FormData();
    formData.append('image', file);
    formData.append('sensitivity', sensitivity);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Sunucu yanıt vermedi.');
      }

      if (!data.result) {
        throw new Error('Sonuç görseli alınamadı.');
      }

      setResultSrc(data.result);
      setMaskSrc(data.mask || null);
      setMessage(data.message || 'İşlem tamamlandı.');
    } catch (err) {
      setError(err.message || 'İşlem başarısız oldu. Backend çalışıyor mu kontrol edin.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadResult = async () => {
    if (!resultSrc) return;

    try {
      setError('');
      const response = await fetch(resultSrc, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error('Görsel indirilemedi.');
      }

      const blob = await response.blob();
      const extension = blob.type.includes('jpeg') ? 'jpg' : blob.type.includes('png') ? 'png' : 'png';
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `cleaned-watermark-image.${extension}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      setError('Görsel indirilemedi. Sonucu Aç butonuyla yeni sekmede açıp manuel kaydedebilirsiniz.');
    }
  };

  const openResultInNewTab = () => {
    if (resultSrc) {
      window.open(resultSrc, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <div style={styles.app}>
      <nav style={styles.nav}>
        <div style={styles.brand}>WaterMark Demo</div>
        <div style={styles.navActions}>
          <NavButton active={page === 'home'} onClick={() => setPage('home')}>Ana Sayfa</NavButton>
          <NavButton active={page === 'about'} onClick={() => setPage('about')}>Hakkında</NavButton>
        </div>
      </nav>

      <main style={styles.main}>
        {page === 'home' && (
          <section style={styles.heroGrid}>
            <div style={styles.heroTextCard}>
              <span style={styles.badge}>Genel Amaçlı Watermark Remover</span>
              <h1 style={styles.heroTitle}>Görsel Watermark Temizleyici</h1>
              <p style={styles.heroText}>
                Görseli yükle, hassasiyeti seç ve backend tarafında watermark tespiti + inpainting işlemini çalıştır. Sonuç, maske ile birlikte aşağıda gösterilir.
              </p>
              <div style={styles.warningBox}>
                Bu uygulama yalnızca kişisel, izinli veya telif hakkı size ait görseller üzerinde eğitim ve demo amacıyla kullanılmalıdır.
              </div>
            </div>

            <div style={styles.card}>
              <SectionTitle
                eyebrow="Demo Alanı"
                title="Görsel yükle ve işle"
                text="Desteklenen formatlar: JPG, PNG, WEBP. Maksimum dosya boyutu: 5MB."
              />

              <label style={styles.uploadBox}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  style={styles.fileInput}
                />
                <span style={styles.uploadIcon}>⬆</span>
                <strong>Görsel seç</strong>
                <small style={styles.fileName}>{selectedFileLabel}</small>
              </label>

              <div style={styles.controlRow}>
                <div style={styles.selectWrap}>
                  <label htmlFor="sensitivity" style={styles.selectLabel}>Hassasiyet</label>
                  <select
                    id="sensitivity"
                    value={sensitivity}
                    onChange={(event) => setSensitivity(event.target.value)}
                    style={styles.select}
                    disabled={loading}
                  >
                    <option value="low">Düşük</option>
                    <option value="medium">Orta</option>
                    <option value="high">Yüksek</option>
                  </select>
                </div>
              </div>

              {loading && <p style={styles.info}>Görsel işleniyor, lütfen bekleyin...</p>}
              {error && <p style={styles.error}>{error}</p>}
              {message && <p style={styles.success}>{message}</p>}

              <div style={styles.previewGrid}>
                <PreviewPanel
                  label="Önce"
                  src={previewSrc}
                  alt="Yüklenen görsel"
                  emptyText="Görsel önizlemesi burada görünecek"
                />
                <PreviewPanel
                  label="Maske"
                  src={maskSrc}
                  alt="Tespit edilen watermark maskesi"
                  emptyText="Maskeyi görmek için görseli işleyin"
                />
                <PreviewPanel
                  label="Sonra"
                  src={resultSrc}
                  alt="İşlenmiş sonuç"
                  emptyText="İşlem sonucu burada görünecek"
                />
              </div>

              <div style={styles.buttonRow}>
                <button
                  type="button"
                  onClick={handleRemoveWatermark}
                  disabled={!file || loading}
                  style={{
                    ...styles.actionButton,
                    ...(!file || loading ? styles.disabledButton : {}),
                  }}
                >
                  {loading ? 'İşleniyor...' : 'Watermark Kaldır'}
                </button>

                {resultSrc && (
                  <button type="button" onClick={handleDownloadResult} style={styles.secondaryButton}>
                    Temizlenmiş Görseli İndir
                  </button>
                )}

                {resultSrc && (
                  <button type="button" onClick={openResultInNewTab} style={styles.secondaryButton}>
                    Sonucu Aç
                  </button>
                )}

                {(file || previewSrc || resultSrc || maskSrc || error || message) && (
                  <button type="button" onClick={resetState} style={styles.resetButton}>
                    Sıfırla
                  </button>
                )}
              </div>
            </div>
          </section>
        )}

        {page === 'about' && (
          <section style={styles.cardWide}>
            <SectionTitle
              eyebrow="Proje Hakkında"
              title="WaterMark Demo"
              text="Bu uygulama, genel amaçlı watermark kaldırma akışını göstermeye yönelik bir frontend + backend prototipidir."
            />
            <div style={styles.aboutTextBlock}>
              <p style={styles.aboutText}>
                Backend tarafı Flask ve OpenCV kullanır. Görselden watermark aday bölgeleri için bir maske üretir ve bu maskeyi inpainting/document cleanup ile temizler. Sonuç görseli ve maske arayüzde gösterilir.
              </p>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

const styles = {
  app: {
    minHeight: '100vh',
    background: '#eef2f7',
    color: '#13203c',
    fontFamily: 'Inter, Arial, sans-serif',
  },
  nav: {
    height: 74,
    padding: '0 24px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    background: '#ffffff',
    borderBottom: '1px solid #dbe3f0',
    position: 'sticky',
    top: 0,
    zIndex: 5,
  },
  brand: {
    fontSize: 18,
    fontWeight: 800,
  },
  navActions: {
    display: 'flex',
    gap: 12,
    flexWrap: 'wrap',
  },
  navButton: {
    border: '1px solid #cad4e5',
    background: '#f7f9fc',
    color: '#30415f',
    padding: '10px 18px',
    borderRadius: 999,
    cursor: 'pointer',
    fontWeight: 700,
  },
  navButtonActive: {
    background: '#1c2d4b',
    color: '#ffffff',
    borderColor: '#1c2d4b',
  },
  main: {
    padding: 32,
    maxWidth: 1320,
    margin: '0 auto',
  },
  heroGrid: {
    display: 'grid',
    gridTemplateColumns: '0.9fr 1.4fr',
    gap: 24,
    alignItems: 'start',
  },
  heroTextCard: {
    background: '#ffffff',
    borderRadius: 28,
    border: '1px solid #d6ddeb',
    padding: 36,
    boxShadow: '0 10px 28px rgba(21, 38, 74, 0.06)',
  },
  badge: {
    display: 'inline-flex',
    alignItems: 'center',
    borderRadius: 999,
    background: '#dfeafc',
    color: '#1d4ed8',
    fontWeight: 800,
    fontSize: 14,
    padding: '8px 14px',
    marginBottom: 18,
  },
  heroTitle: {
    fontSize: 52,
    lineHeight: 1.04,
    margin: '0 0 20px',
    letterSpacing: '-0.03em',
  },
  heroText: {
    fontSize: 16,
    lineHeight: 1.8,
    color: '#47597d',
    margin: 0,
  },
  warningBox: {
    marginTop: 28,
    border: '1px solid #f0bb82',
    background: '#fff6ec',
    color: '#b45309',
    padding: 18,
    borderRadius: 18,
    lineHeight: 1.6,
    fontWeight: 600,
  },
  card: {
    background: '#ffffff',
    borderRadius: 28,
    border: '1px solid #d6ddeb',
    padding: 28,
    boxShadow: '0 10px 28px rgba(21, 38, 74, 0.06)',
  },
  cardWide: {
    background: '#ffffff',
    borderRadius: 28,
    border: '1px solid #d6ddeb',
    padding: 30,
    boxShadow: '0 10px 28px rgba(21, 38, 74, 0.06)',
  },
  sectionHeader: {
    marginBottom: 20,
  },
  eyebrow: {
    display: 'inline-block',
    textTransform: 'uppercase',
    color: '#335eea',
    fontWeight: 800,
    letterSpacing: '0.08em',
    fontSize: 13,
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 28,
    lineHeight: 1.1,
    margin: '0 0 10px',
  },
  sectionText: {
    margin: 0,
    color: '#576987',
    lineHeight: 1.7,
  },
  uploadBox: {
    border: '2px dashed #9fc0ff',
    background: '#f5f9ff',
    borderRadius: 24,
    minHeight: 132,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'column',
    gap: 8,
    cursor: 'pointer',
    textAlign: 'center',
    padding: 20,
  },
  fileInput: {
    display: 'none',
  },
  uploadIcon: {
    width: 40,
    height: 40,
    borderRadius: 14,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#335eea',
    color: '#fff',
    fontSize: 22,
    fontWeight: 800,
  },
  fileName: {
    color: '#62718a',
    wordBreak: 'break-word',
  },
  controlRow: {
    display: 'flex',
    gap: 16,
    marginTop: 18,
    marginBottom: 8,
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  selectWrap: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  selectLabel: {
    fontSize: 14,
    color: '#42526c',
    fontWeight: 700,
  },
  select: {
    minWidth: 170,
    padding: '12px 14px',
    borderRadius: 14,
    border: '1px solid #ccd6e7',
    background: '#fff',
    color: '#13203c',
  },
  info: {
    marginTop: 16,
    background: '#eff6ff',
    border: '1px solid #bfdbfe',
    color: '#1d4ed8',
    padding: '12px 14px',
    borderRadius: 14,
    lineHeight: 1.5,
  },
  error: {
    marginTop: 16,
    background: '#fff1f2',
    border: '1px solid #fecdd3',
    color: '#be123c',
    padding: '12px 14px',
    borderRadius: 14,
    lineHeight: 1.5,
  },
  success: {
    marginTop: 16,
    background: '#effdf4',
    border: '1px solid #b7ebc6',
    color: '#15803d',
    padding: '12px 14px',
    borderRadius: 14,
    lineHeight: 1.5,
  },
  previewGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
    gap: 18,
    marginTop: 20,
  },
  previewPanel: {
    borderRadius: 22,
    border: '1px solid #dbe3f0',
    background: '#f8fafc',
    padding: 14,
  },
  label: {
    margin: '0 0 10px',
    fontWeight: 800,
    fontSize: 15,
  },
  image: {
    width: '100%',
    height: 280,
    objectFit: 'contain',
    borderRadius: 16,
    background: '#ffffff',
  },
  emptyPreview: {
    height: 280,
    borderRadius: 16,
    background: '#e9eef6',
    color: '#62718a',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
    padding: 20,
  },
  buttonRow: {
    display: 'flex',
    gap: 12,
    flexWrap: 'wrap',
    marginTop: 20,
  },
  actionButton: {
    border: 'none',
    borderRadius: 16,
    background: '#335eea',
    color: '#ffffff',
    padding: '14px 22px',
    fontWeight: 800,
    cursor: 'pointer',
    fontSize: 15,
  },
  secondaryButton: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    textDecoration: 'none',
    borderRadius: 16,
    background: '#eef3ff',
    color: '#2343a7',
    padding: '14px 22px',
    fontWeight: 800,
    border: '1px solid #c8d7ff',
    cursor: 'pointer',
  },
  resetButton: {
    borderRadius: 16,
    background: '#ffffff',
    color: '#475569',
    padding: '14px 22px',
    fontWeight: 800,
    border: '1px solid #cbd5e1',
    cursor: 'pointer',
  },
  disabledButton: {
    opacity: 0.55,
    cursor: 'not-allowed',
  },
  aboutTextBlock: {
    background: '#f8fafc',
    border: '1px solid #dbe3f0',
    borderRadius: 20,
    padding: 20,
  },
  aboutText: {
    margin: 0,
    color: '#576987',
    lineHeight: 1.8,
  },
};
