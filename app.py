# -*- coding: utf-8 -*-
"""
Öğrenci Terk (Dropout) Riski Erken Uyarı API'si  -  Türkiye Uyarlaması
=======================================================================

Mimari
------
FastAPI + Pydantic v2. Streamlit arayüzünün yerine, başka sistemlerin (OBS, CRM,
danışmanlık paneli) çağırabileceği bir REST servisi sunar.

Dosyalar (aynı klasörde ya da MODEL_DIR ortam değişkeninin gösterdiği klasörde):
    rf_model.pkl, scaler.pkl, model_columns.pkl

Çalıştırma
----------
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000
    # Üretim: gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 app:app

Endpoint'ler
------------
    GET  /health              Liveness  (her zaman 200; durumu raporlar)
    GET  /ready               Readiness (model yüklü değilse 503)
    GET  /template            Örnek JSON payload + alan açıklamaları
    POST /predict             Tek öğrenci tahmini
    POST /predict/batch       Çoklu öğrenci (bozuk kayıt diğerlerini düşürmez)
    GET  /feature-importance  Random Forest özellik önemi + Türkiye bağlamı
    GET  /model-info          Model künyesi ve aşırı öğrenme teşhisi

!!! ÇOK ÖNEMLİ - MODELİN GEÇERLİLİK SINIRI !!!
---------------------------------------------
`rf_model.pkl`, UCI "Predict Students' Dropout and Academic Success" veri setiyle
(Portekiz, Portalegre Politeknik Enstitüsü, 2008-2019) eğitilmiştir. Bu nedenle:

  1. Modelde KYK, aylık harçlık dengesi, bölüm memnuniyeti ve kısmi zamanlı
     çalışma diye bir sütun YOKTUR. Bu bilgiler API'de kabul edilir, doğrulanır
     ve "danışman uyarıları" (advisory) olarak ayrıca raporlanır; ancak modelin
     olasılıklarını DEĞİŞTİRMEZ. Çünkü eğitim verisinde karşılığı olmayan bir
     özelliğe uydurma bir katsayı vermek, bilimsel olarak savunulamaz. Bu
     özelliklerin modele girmesi için Türkiye verisiyle yeniden eğitim gerekir.
  2. Türkiye'deki alanlar, modelin beklediği sütunlara AÇIKÇA belgelenmiş
     dönüşümlerle eşlenir (bkz. `map_payload_to_features`). Eşlenemeyen sütunlar
     (Portekiz kodlu Course, meslek, ebeveyn eğitimi, uyruk, makroekonomi)
     referans kategoride / eğitim ortalamasında sabitlenir. Bu yüzden çıktılar
     "Türkiye için kalibre edilmiş kesin olasılık" değil, "Portekiz-eğitimli
     modelin Türkiye profiline projeksiyonu"dur.
  3. Üretimde karar vermeden önce kurumun kendi geçmiş kohortunda (mezun/terk
     edilmiş gerçek etiketlerle) geriye dönük doğrulama yapılması şarttır.

Kişisel Veri (KVKK)
-------------------
Bu servis öğrenci verisini diske yazmaz ve payload'ı loglamaz; yalnızca istek
kimliği, yol, durum kodu ve süre loglanır. Üst düzey `ogrenci_ref` alanına TC
kimlik no / öğrenci no yerine takma (pseudonymous) bir kimlik gönderin.
Cinsiyet modelde yer alan bir özelliktir; kararların cinsiyete göre farklılaşması
adalet (fairness) açısından kurum içinde ayrıca değerlendirilmelidir.

AŞIRI ÖĞRENME (OVERFITTING) - EN İYİ PRATİKLER  (modelin YENİDEN EĞİTİMİ için)
----------------------------------------------------------------------------
"Sıfır hata" hedefi yanlış bir hedeftir: eğitim kümesinde %100 doğruluk,
ezberlemenin işaretidir. Yüklü modelde min_samples_leaf=1, max_depth=None,
300 ağaç, gözlenen derinlik ~45 -> ağaçlar eğitim verisini ezberleyecek kadar
derindir (62 MB'lık dosya boyutu da bunun yan etkisidir). Önerilenler:

  * Hedef: eğitim değil, ZAMAN BAZLI doğrulama kümesindeki hata. Rastgele
    train/test yerine kayıt yılına göre ayırın (geçmiş yıllar -> eğitim, son yıl
    -> test); öğrenci düzeyinde sızıntı (data leakage) olmasın.
  * Hiperparametre kısıtı: min_samples_leaf 5-20, max_depth 8-16,
    max_features="sqrt", n_estimators 300-500, max_samples 0.7-0.8,
    class_weight="balanced_subsample" (Dropout sınıfı azınlıksa).
  * Stratified K-Fold (k=5) çapraz doğrulama; eğitim-doğrulama farkı (gap)
    izlenmeli. oob_score=True ücretsiz bir ek genelleme tahminidir.
  * Metrik: accuracy tek başına yetmez; Dropout sınıfı için recall/F1,
    macro-F1, PR-AUC; olasılıklar için Brier skoru ve kalibrasyon eğrisi.
  * Olasılık kalibrasyonu: CalibratedClassifierCV(method="isotonic") ya da
    "sigmoid"; erken uyarı eşiği (örn. 0.30) kalibre olasılık üzerinde seçilmeli.
  * Sızıntı kontrolü: 2. dönem sonu sütunları, dönem sonu bilinmeyen bir
    kararı (örn. mezuniyet) dolaylı kodluyor olabilir; tahmin anına göre hangi
    verinin gerçekten mevcut olduğunu doğrulayın.
  * Özellik önemi için MDI (Gini) yerine, ayrı bir doğrulama kümesinde
    permutation importance / SHAP kullanın. MDI, sürekli ve yüksek kardinaliteli
    değişkenleri kayırır; korelasyonlu değişkenlerin önemini böler.
  * Türkiye verisi toplandığında: Course/bölüm, üniversite türü (devlet/vakıf),
    öğretim türü, KYK/burs, çalışma durumu gibi değişkenleri doğrudan ekleyin;
    YKS puanını kohort içi yüzdelik dilime çevirin; bölüm/şehir için grup
    bazlı (GroupKFold) doğrulama kullanın.
  * Model izleme: üretimde girdi dağılımı kayması (drift) ve gerçekleşen
    sonuçlarla periyodik karşılaştırma yapılmalı; model sürümü kayıtlı olmalı.
"""

from __future__ import annotations

import difflib
import logging
import math
import os
import re
import threading
import time
import unicodedata
import uuid
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)
from starlette.exceptions import HTTPException as StarletteHTTPException

# =============================================================================
# 1. YAPILANDIRMA (ortam değişkenleri) VE LOGLAMA
# =============================================================================


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    """Ortam değişkenini güvenle float'a çevirir; bozuksa varsayılana döner."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw.replace(",", "."))
        return value if math.isfinite(value) else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent)))
MODEL_FILE, SCALER_FILE, COLUMNS_FILE = "rf_model.pkl", "scaler.pkl", "model_columns.pkl"


HIGH_RISK_THRESHOLD = _env_float("HIGH_RISK_THRESHOLD", 0.50)
MEDIUM_RISK_THRESHOLD = _env_float("MEDIUM_RISK_THRESHOLD", 0.30)
MAX_BATCH_SIZE = _env_int("MAX_BATCH_SIZE", 200)
RELOAD_COOLDOWN_SEC = _env_int("RELOAD_COOLDOWN_SEC", 30)
OOD_Z_THRESHOLD = _env_float("OOD_Z_THRESHOLD", 4.0)


YKS_COHORT_MEAN = _env_float("YKS_COHORT_MEAN", None)
YKS_COHORT_STD = _env_float("YKS_COHORT_STD", None)
LISE_COHORT_MEAN = _env_float("LISE_COHORT_MEAN", None)  # diploma notu, 0-100
LISE_COHORT_STD = _env_float("LISE_COHORT_STD", None)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("dropout-api")

FALLBACK_NUMERIC_COLS = [
    "Application order",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]
CLASS_NAMES_DEFAULT = ["Dropout", "Enrolled", "Graduate"]  # LabelEncoder (alfabetik) sırası
CLASS_LABELS_TR = {
    "Dropout": "Terk (Dropout)",
    "Enrolled": "Kayıtlı - Devam Ediyor (Enrolled)",
    "Graduate": "Mezun (Graduate)",
}
RISK_LABELS_TR = {"dusuk": "Düşük risk", "orta": "Orta risk", "yuksek": "Yüksek risk"}

DISCLAIMER_TR = (
    "Bu çıktı, UCI/Portekiz verisiyle eğitilmiş bir modelin Türkiye profiline "
    "yaklaşık projeksiyonudur; kalibre edilmiş bir olasılık veya kesin karar değildir. "
    "KYK, harçlık, memnuniyet ve çalışma bilgileri yalnızca danışman uyarısı üretir, "
    "model olasılıklarını değiştirmez. Karar süreçlerinde yalnızca erken uyarı "
    "göstergesi olarak, insan değerlendirmesiyle birlikte kullanın."
)


# =============================================================================
# 2. HATA SINIFLARI
# =============================================================================


class ApiError(Exception):
    """Kontrollü, istemciye anlamlı mesaj döndürülen uygulama hatası."""

    def __init__(self, status_code: int, code: str, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


class ArtifactError(ApiError):
    """Model/scaler/kolon dosyaları yüklenemedi ya da birbiriyle uyumsuz (503)."""

    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(503, "model_unavailable", message, details)


class PreprocessingError(ApiError):
    """Girdi, modelin beklediği forma güvenle dönüştürülemedi (422)."""

    def __init__(self, code: str, message: str, details: Optional[Any] = None) -> None:
        super().__init__(422, code, message, details)


# =============================================================================
# 3. ARTEFAKTLARIN (MODEL / SCALER / KOLONLAR) YÜKLENMESİ VE DOĞRULANMASI
# =============================================================================


@dataclass
class Artifacts:
    """Doğrulanmış, kullanıma hazır model bileşenleri."""

    model: Any
    scaler: Any
    columns: list[str]
    numeric_cols: list[str]
    scaler_mean: dict[str, float]
    scaler_scale: dict[str, float]
    class_names: list[str]
    importances: pd.Series  # özellik -> MDI önemi
    tree_importances: np.ndarray  # (n_ağaç, n_özellik)
    model_stats: dict[str, Any]
    load_warnings: list[str] = field(default_factory=list)
    loaded_at: float = field(default_factory=time.time)


def _resolve_class_names(model: Any) -> list[str]:
    """model.classes_ -> ['Dropout','Enrolled','Graduate'] eşlemesini doğrular."""
    classes = list(getattr(model, "classes_", []))
    if len(classes) != 3:
        raise ArtifactError(
            f"Model 3 sınıf bekleniyordu, {len(classes)} sınıf bulundu.", {"classes": [str(c) for c in classes]}
        )
    names: list[str] = []
    for cls in classes:
        if isinstance(cls, (int, np.integer)) and 0 <= int(cls) < 3:
            names.append(CLASS_NAMES_DEFAULT[int(cls)])
        elif isinstance(cls, str) and cls in CLASS_NAMES_DEFAULT:
            names.append(cls)
        else:
            raise ArtifactError(
                "Bilinmeyen sınıf etiketi; LabelEncoder eşlemesi doğrulanamadı.", {"classes": [str(c) for c in classes]}
            )
    if len(set(names)) != 3:
        raise ArtifactError("Sınıf etiketleri benzersiz değil.", {"classes": names})
    return names


def _load_pickle(path: Path) -> tuple[Any, list[str]]:
    """Bir pickle dosyasını yükler; sürüm uyuşmazlığı uyarılarını toplar."""
    if not path.is_file():
        raise ArtifactError(f"Dosya bulunamadı: {path.name}", {"aranan_klasor": str(path.parent)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            obj = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - bozuk/uyumsuz pickle her türlü hatayı verebilir
            raise ArtifactError(
                f"{path.name} yüklenemedi (bozuk dosya ya da scikit-learn sürüm uyumsuzluğu).",
                {"hata": f"{type(exc).__name__}: {exc}"},
            ) from exc
    return obj, [f"{path.name}: {w.message}" for w in caught]


def load_artifacts(model_dir: Path = MODEL_DIR) -> Artifacts:
    """3 dosyayı yükler ve birbirleriyle UYUMLULUĞUNU doğrular.

    Kontroller: sütun listesi tekrarsız mı; model.n_features_in_ == len(sütunlar);
    modelin feature_names_in_ değeri sütun sırasıyla aynı mı; scaler sütunları
    model sütunlarının alt kümesi mi; sınıf eşlemesi; uçtan uca duman testi.
    """
    load_warnings: list[str] = []

    model, w1 = _load_pickle(model_dir / MODEL_FILE)
    scaler, w2 = _load_pickle(model_dir / SCALER_FILE)
    columns_obj, w3 = _load_pickle(model_dir / COLUMNS_FILE)
    load_warnings += w1 + w2 + w3

    # --- sütun listesi ---
    if not isinstance(columns_obj, (list, tuple, pd.Index, np.ndarray)):
        raise ArtifactError("model_columns.pkl bir sütun listesi içermiyor.", {"tip": type(columns_obj).__name__})
    columns = [str(c) for c in list(columns_obj)]
    if len(columns) != len(set(columns)):
        dup = sorted({c for c in columns if columns.count(c) > 1})
        raise ArtifactError("model_columns.pkl içinde yinelenen sütun adları var.", {"tekrar": dup})

    # --- model ---
    if not hasattr(model, "predict_proba") or not hasattr(model, "estimators_"):
        raise ArtifactError(
            "rf_model.pkl beklenen bir RandomForest sınıflandırıcısı değil.", {"tip": type(model).__name__}
        )
    if int(model.n_features_in_) != len(columns):
        raise ArtifactError(
            "Model ile model_columns.pkl sütun sayısı uyuşmuyor.",
            {"model": int(model.n_features_in_), "model_columns": len(columns)},
        )
    model_names = getattr(model, "feature_names_in_", None)
    if model_names is not None and [str(c) for c in model_names] != columns:
        raise ArtifactError("Modelin eğitim sütun sırası model_columns.pkl ile aynı değil.")
    # Tek satırlık isteklerde n_jobs=-1 gereksiz iş parçacığı yaratır ve eşzamanlı
    # isteklerde CPU'yu aşırı abone eder; istek başına tek iş parçacığı yeterlidir.
    try:
        model.set_params(n_jobs=1)
    except Exception:  # noqa: BLE001
        logger.warning("model.n_jobs ayarlanamadı; varsayılan ile devam ediliyor.")

    # --- scaler ---
    if not hasattr(scaler, "transform") or not hasattr(scaler, "mean_"):
        raise ArtifactError("scaler.pkl fit edilmiş bir StandardScaler değil.", {"tip": type(scaler).__name__})
    scaler_names = getattr(scaler, "feature_names_in_", None)
    numeric_cols = [str(c) for c in scaler_names] if scaler_names is not None else list(FALLBACK_NUMERIC_COLS)
    if len(numeric_cols) != int(scaler.n_features_in_):
        raise ArtifactError(
            "Scaler sütun sayısı beklenenle uyuşmuyor.",
            {"scaler": int(scaler.n_features_in_), "beklenen": len(numeric_cols)},
        )
    missing = [c for c in numeric_cols if c not in columns]
    if missing:
        raise ArtifactError("Scaler sütunları model_columns.pkl içinde bulunamadı.", {"eksik": missing})
    scale = np.where(np.asarray(scaler.scale_, dtype=float) == 0, 1.0, scaler.scale_)
    mean_map = {c: float(m) for c, m in zip(numeric_cols, scaler.mean_)}
    scale_map = {c: float(s) for c, s in zip(numeric_cols, scale)}

    class_names = _resolve_class_names(model)

    # --- özellik önemi ve aşırı öğrenme teşhisi (bir kez hesaplanır) ---
    importances = pd.Series(model.feature_importances_, index=columns, dtype=float)
    tree_imp = np.vstack([t.feature_importances_ for t in model.estimators_])
    depths = [int(e.tree_.max_depth) for e in model.estimators_]
    nodes = [int(e.tree_.node_count) for e in model.estimators_]
    stats = {
        "n_estimators": len(model.estimators_),
        "max_depth_param": model.max_depth,
        "min_samples_leaf_param": model.min_samples_leaf,
        "observed_max_depth": max(depths),
        "observed_mean_depth": round(float(np.mean(depths)), 1),
        "mean_nodes_per_tree": round(float(np.mean(nodes)), 0),
        "oob_score_enabled": bool(getattr(model, "oob_score", False)),
        "class_weight": None if model.class_weight is None else str(model.class_weight),
    }

    art = Artifacts(
        model,
        scaler,
        columns,
        numeric_cols,
        mean_map,
        scale_map,
        class_names,
        importances,
        tree_imp,
        stats,
        load_warnings,
    )

    # --- uçtan uca duman testi: pipeline gerçekten çalışıyor mu? ---
    try:
        frame, _ = build_model_frame(art, {})
        proba = model.predict_proba(frame)
        if proba.shape != (1, 3) or not np.isfinite(proba).all() or abs(float(proba.sum()) - 1.0) > 1e-6:
            raise ValueError(f"Beklenmeyen çıktı şekli/değeri: {proba!r}")
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(
            "Duman testi başarısız: model ve ön işleme uyumsuz.", {"hata": f"{type(exc).__name__}: {exc}"}
        ) from exc
    return art


class ArtifactStore:
    """Artefaktları iş parçacığı güvenli tutar; yükleme başarısızsa servis AYAKTA kalır.

    Servis model dosyaları olmadan da başlar (degraded). İstekler 503 alır ve
    dosyalar sonradan yerleştirilirse, bekleme süresi (cooldown) dolunca otomatik
    yeniden yükleme denenir. Böylece süreç çökmez, yeniden başlatma gerekmez.
    """

    def __init__(self) -> None:
        self._art: Optional[Artifacts] = None
        self._error: Optional[ApiError] = None
        self._last_attempt = 0.0
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            self._last_attempt = time.time()
            try:
                self._art = load_artifacts()
                self._error = None
                logger.info("Model artefaktları yüklendi (%d sütun).", len(self._art.columns))
                for msg in self._art.load_warnings:
                    logger.warning("Yükleme uyarısı: %s", msg)
            except ApiError as exc:
                self._art, self._error = None, exc
                logger.error("Artefakt yükleme hatası: %s | %s", exc.message, exc.details)
            except Exception as exc:  # noqa: BLE001
                self._art = None
                self._error = ArtifactError("Beklenmeyen yükleme hatası.", {"hata": f"{type(exc).__name__}: {exc}"})
                logger.exception("Beklenmeyen artefakt yükleme hatası")

    def get(self) -> Artifacts:
        if self._art is not None:
            return self._art
        if time.time() - self._last_attempt >= RELOAD_COOLDOWN_SEC:
            self.load()
            if self._art is not None:
                return self._art
        raise self._error or ArtifactError("Model henüz yüklenmedi.")

    @property
    def error(self) -> Optional[ApiError]:
        return self._error

    @property
    def ready(self) -> bool:
        return self._art is not None


store = ArtifactStore()


# =============================================================================
# 4. TÜRKİYE'YE ÖZGÜ DÖNÜŞÜM YARDIMCILARI VE ESNEK TİP DÖNÜŞTÜRÜCÜLER
# =============================================================================

_TR_TRANSLATE = str.maketrans(
    {"ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c", "â": "a", "î": "i", "û": "u"}
)


def _norm_text(value: Any) -> Any:
    """'Vakıf', 'VAKIF ', 'Kadın' -> 'vakif', 'kadin' (Türkçe büyük/küçük harf tuzaklarına karşı)."""
    if not isinstance(value, str):
        return value
    text = value.strip().replace("İ", "i").replace("I", "ı").lower()
    text = text.translate(_TR_TRANSLATE)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[\s\-]+", "_", text)


def _coerce_float(value: Any) -> float:
    """Sayıyı esnek biçimde okur: 12.5, "12,5" (Türkçe ondalık), "13" kabul; NaN/Inf/bool red."""
    if isinstance(value, bool):
        raise ValueError("Sayı bekleniyor, mantıksal (true/false) değer geldi")
    if isinstance(value, str):
        value = value.strip().replace(" ", "").replace("%", "")
        if value.count(",") == 1 and "." not in value:
            value = value.replace(",", ".")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('Sayı bekleniyor (örn. 12.5 ya da "12,5")') from None
    if not math.isfinite(number):
        raise ValueError("Sonlu bir sayı bekleniyor (NaN/Infinity kabul edilmez)")
    return number


def _coerce_int(value: Any) -> int:
    number = _coerce_float(value)
    if abs(number - round(number)) > 1e-9:
        raise ValueError("Tam sayı bekleniyor")
    return int(round(number))


_TRUE = {"evet", "e", "true", "yes", "y", "1", "var", "dogru"}
_FALSE = {"hayir", "h", "false", "no", "n", "0", "yok", "yanlis"}


def _coerce_bool(value: Any) -> bool:
    """true/false, 1/0, "evet"/"hayır", "var"/"yok" değerlerini kabul eder."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = _norm_text(value)
        if token in _TRUE:
            return True
        if token in _FALSE:
            return False
    raise ValueError("Mantıksal değer bekleniyor (true/false, 1/0 veya evet/hayır)")


LFloat = Annotated[float, BeforeValidator(_coerce_float)]
LInt = Annotated[int, BeforeValidator(_coerce_int)]
LBool = Annotated[bool, BeforeValidator(_coerce_bool)]


def _lit(*options: str) -> Any:
    """Normalize edilmiş (küçük harf, ASCII) bir Literal alan tipi üretir."""
    return Annotated[Literal[options], BeforeValidator(_norm_text)]


Cinsiyet = _lit("erkek", "kadin")
Uyruk = _lit("tc", "yabanci")
UniversiteTuru = _lit("devlet", "vakif")
OgretimTuru = _lit("birinci", "ikinci")
KykDurumu = _lit("yok", "kredi", "burs", "kredi_ve_burs")
Barinma = _lit("aile_yaninda", "kyk_yurdu", "ozel_yurt", "kirada")


def cohort_to_model_scale(
    value: float, cohort_mean: float, cohort_std: float, train_mean: float, train_std: float, clip: float = 3.0
) -> float:
    """Kohort içi z-skoru hesaplayıp eğitim dağılımına taşır (z ±3'te kırpılır).

    Örnek: kurumun YKS ortalaması 380, std 55 olsun. 435 puanlı öğrenci z=+1 ->
    modele "Portekiz giriş notu ortalamasının 1 std üstü" olarak girer.
    """
    z = (value - cohort_mean) / cohort_std
    z = max(-clip, min(clip, z))
    return train_mean + z * train_std


# =============================================================================
# 5. GİRDİ ŞEMASI (Pydantic) - ESNEK AMA DOĞRULANMIŞ JSON PAYLOAD
# =============================================================================


class _Strict(BaseModel):
    """Tanımsız alanları reddeder: yazım hataları ('gecilen_dres') sessizce yutulmaz."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SemesterRecord(_Strict):
    """Bir yarıyılın akademik özeti (vize-final ağırlıklı sistem)."""

    alinan_ders: LInt = Field(ge=0, le=30, description="Dönem başında kayıt olunan ders sayısı")
    gecilen_ders: LInt = Field(ge=0, le=30, description="Başarıyla geçilen (DD ve üzeri) ders sayısı")
    sinav_sayisi: Optional[LInt] = Field(
        default=None,
        ge=0,
        le=80,
        description="Girilen sınav sayısı (vize+final+bütünleme...). Yoksa eğitim oranıyla tahmin edilir.",
    )
    muaf_ders: LInt = Field(
        default=0, ge=0, le=30, description="Muafiyet / yatay geçiş / intibak ile sayılan ders sayısı"
    )
    devamsiz_ders: LInt = Field(
        default=0, ge=0, le=30, description="Devamsızlıktan kalan / değerlendirmeye girmeyen ders sayısı"
    )
    agno: Optional[LFloat] = Field(default=None, ge=0, le=4, description="Dönem not ortalaması, 4'lük sistem (0-4)")
    vize_ortalamasi: Optional[LFloat] = Field(
        default=None, ge=0, le=100, description="Vize ortalaması (0-100); agno yoksa kullanılır"
    )
    final_ortalamasi: Optional[LFloat] = Field(
        default=None, ge=0, le=100, description="Final ortalaması (0-100); agno yoksa kullanılır"
    )
    vize_agirligi: LFloat = Field(
        default=0.40, ge=0, le=1, description="Vize ağırlığı (varsayılan 0.40; final = 1 - vize)"
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> "SemesterRecord":
        if self.gecilen_ders > self.alinan_ders + self.muaf_ders:
            raise ValueError("gecilen_ders, alinan_ders + muaf_ders toplamını aşamaz")
        if self.devamsiz_ders > self.alinan_ders:
            raise ValueError("devamsiz_ders, alinan_ders değerini aşamaz")
        has_pair = self.vize_ortalamasi is not None and self.final_ortalamasi is not None
        if self.agno is None and not has_pair:
            raise ValueError("Not bilgisi eksik: 'agno' ya da ('vize_ortalamasi' + 'final_ortalamasi') gönderin")
        return self

    def grade_on_20_scale(self) -> float:
        """Not ortalamasını modelin 0-20 ölçeğine çevirir (4'lük x5; 100'lük /5).

        AGNO 2.00 (geçme sınırı) -> 10/20: Portekiz sistemindeki geçme notuyla örtüşür.
        """
        if self.agno is not None:
            return self.agno * 5.0
        weighted = self.vize_ortalamasi * self.vize_agirligi + self.final_ortalamasi * (
            1.0 - self.vize_agirligi
        )  # type: ignore[operator]
        return weighted / 5.0


class AcademicSection(_Strict):
    donem_1: SemesterRecord = Field(description="Güz yarıyılı")
    donem_2: SemesterRecord = Field(description="Bahar yarıyılı")


class StudentProfile(_Strict):
    kayit_yasi: LInt = Field(ge=15, le=80, description="Üniversiteye kayıt olduğu andaki yaş")
    cinsiyet: Cinsiyet = Field(description="'erkek' | 'kadin' (modelin girdisidir)")
    uyruk: Uyruk = Field(default="tc", description="'tc' | 'yabanci' (yabancı uyruklu öğrenci)")
    ozel_egitim_ihtiyaci: LBool = Field(default=False, description="Engelli/özel eğitim desteği ihtiyacı")
    ailesinden_farkli_sehirde: LBool = Field(
        default=False, description="Okuduğu şehir ailenin ikamet şehrinden farklı mı? (KYK yurt ihtiyacı göstergesi)"
    )


class AdmissionSection(_Strict):
    universite_turu: UniversiteTuru = Field(default="devlet", description="'devlet' | 'vakif'")
    ogretim_turu: OgretimTuru = Field(default="birinci", description="'birinci' (gündüz) | 'ikinci' (akşam)")
    yks_yerlestirme_puani: Optional[LFloat] = Field(
        default=None, ge=100, le=560, description="YKS yerleştirme puanı (OBP dahil, 100-560)"
    )
    admission_grade_0_200: Optional[LFloat] = Field(
        default=None,
        ge=0,
        le=200,
        description="(İleri düzey) Modelin ölçeğinde hazır giriş notu; verilirse YKS dönüşümünü ezer",
    )
    lise_diploma_notu: Optional[LFloat] = Field(default=None, ge=50, le=100, description="Lise diploma notu (50-100)")
    tercih_sirasi: LInt = Field(default=1, ge=1, le=30, description="YKS tercih sırası (1 = ilk tercih)")


class FinancialSection(_Strict):
    """Türkiye'ye özgü mali/sosyal alanlar. Hepsi isteğe bağlıdır."""

    kyk_durumu: KykDurumu = Field(default="yok", description="KYK: 'yok' | 'kredi' | 'burs' | 'kredi_ve_burs'")
    barinma: Barinma = Field(
        default="aile_yaninda", description="Barınma: 'aile_yaninda' | 'kyk_yurdu' | 'ozel_yurt' | 'kirada'"
    )
    diger_burs: LBool = Field(default=False, description="KYK dışı burs (üniversite/vakıf/kurum/kamu) var mı?")
    vakif_burs_orani: LFloat = Field(
        default=0.0, ge=0, le=100, description="Vakıf üniversitesinde burs oranı (%100, %50, %25...; 0 = ücretli)"
    )
    harc_odemesi_guncel: Optional[LBool] = Field(
        default=None, description="Harç/öğrenim ücreti güncel mi? (vakıf ve ikinci öğretimde anlamlıdır)"
    )
    borclu: LBool = Field(default=False, description="Vadesi geçmiş okul/yurt borcu var mı?")
    aylik_gelir_try: Optional[LFloat] = Field(
        default=None, ge=0, le=10_000_000, description="Aylık toplam gelir/harçlık (TL)"
    )
    aylik_gider_try: Optional[LFloat] = Field(
        default=None, ge=0, le=10_000_000, description="Aylık zorunlu gider (barınma+yemek+ulaşım...) (TL)"
    )
    aylik_butce_dengesi_try: Optional[LFloat] = Field(
        default=None,
        ge=-10_000_000,
        le=10_000_000,
        description="Doğrudan denge (gelir - gider, TL). Negatif = açık. gelir/gider verilmişse onlar öncelikli.",
    )
    kismi_zamanli_calisiyor: LBool = Field(default=False, description="Kısmi zamanlı/ek iş")
    haftalik_calisma_saati: Optional[LFloat] = Field(default=None, ge=0, le=80, description="Haftalık çalışma saati")


class SurveySection(_Strict):
    bolum_memnuniyeti: Optional[LInt] = Field(
        default=None, ge=1, le=5, description="Bölüm memnuniyeti (1 = çok memnuniyetsiz ... 5 = çok memnun)"
    )


class StudentPayload(_Strict):
    """Tek öğrenci için tam istek gövdesi."""

    ogrenci_ref: Optional[str] = Field(
        default=None, max_length=64, description="Takma kimlik (TC/öğrenci no yerine). Yanıtta aynen döner."
    )
    ogrenci: StudentProfile
    akademik: AcademicSection
    giris: AdmissionSection = Field(default_factory=AdmissionSection)
    mali: FinancialSection = Field(default_factory=FinancialSection)
    anket: SurveySection = Field(default_factory=SurveySection)
    raw_features: Optional[dict[str, LFloat]] = Field(
        default=None,
        description="(İleri düzey) Model sütun adlarıyla doğrudan değer (örn. {'Course_9500': 1}). "
        "Dönüşümle üretilen değerleri ezer; sütun adları doğrulanır.",
    )


class BatchRequest(_Strict):
    ogrenciler: list[dict[str, Any]] = Field(min_length=1, description="StudentPayload nesneleri listesi")

    @model_validator(mode="after")
    def _limit(self) -> "BatchRequest":
        if len(self.ogrenciler) > MAX_BATCH_SIZE:
            raise ValueError(f"Tek istekte en fazla {MAX_BATCH_SIZE} öğrenci gönderilebilir")
        return self


# =============================================================================
# 6. TÜRKİYE PAYLOAD'INI MODEL SÜTUNLARINA EŞLEME
# =============================================================================


@dataclass
class MappedInput:
    features: dict[str, float] = field(default_factory=dict)  # ham (ölçeklenmemiş) model girdileri
    trace: dict[str, str] = field(default_factory=dict)  # özellik -> hangi kuralla üretildi
    warnings: list[str] = field(default_factory=list)  # veri kalitesi uyarıları
    assumptions: list[str] = field(default_factory=list)  # yapılan varsayımlar
    derived: dict[str, Any] = field(default_factory=dict)  # danışman kuralları için ara değerler


def map_payload_to_features(art: Artifacts, p: StudentPayload) -> MappedInput:
    """Türkiye'ye özgü JSON'u UCI model sütunlarına çevirir. HER eşleme belgelidir.

    Eşleme kararları (gerekçeli):
      * Scholarship holder   = KYK bursu VEYA diğer burs VEYA vakıf burs oranı > 0.
                               KYK *kredisi* geri ödemeli olduğundan burs sayılmaz.
      * Tuition fees up to date = vakıf üniversitesi veya ikinci öğretimde (harçlı
                               programlar) 'harc_odemesi_guncel'; devlet/birinci
                               öğretimde harç olmadığından 1 (güncel) kabul edilir.
      * Debtor               = 'borclu' (vadesi geçmiş okul/yurt borcu). KYK kredisi
                               borcu mezuniyet sonrası başladığı için sayılmaz.
      * Displaced            = ailesinden_farkli_sehirde (KYK yurt/barınma ihtiyacı).
      * Daytime/evening      = birinci öğretim 1, ikinci öğretim 0.
      * Application order    = tercih_sirasi - 1, [0, 9] aralığına kırpılır.
      * Previous qualification = 1 (lise/orta öğretim); UCI kodlaması.
      * Grades               = AGNO x 5 ya da (vize/final ağırlıklı ortalama) / 5.
      * Gender               = erkek 1 / kadın 0 (UCI sözlüğü).
    """
    m = MappedInput()
    f, tr = m.features, m.trace
    prof, adm, fin = p.ogrenci, p.giris, p.mali

    # ---- demografik / ikili ----
    f["Age at enrollment"] = float(prof.kayit_yasi)
    tr["Age at enrollment"] = "ogrenci.kayit_yasi"
    f["Gender"] = 1.0 if prof.cinsiyet == "erkek" else 0.0
    tr["Gender"] = "ogrenci.cinsiyet (erkek=1, kadin=0)"
    f["International"] = 1.0 if prof.uyruk == "yabanci" else 0.0
    tr["International"] = "ogrenci.uyruk"
    f["Educational special needs"] = float(prof.ozel_egitim_ihtiyaci)
    tr["Educational special needs"] = "ogrenci.ozel_egitim_ihtiyaci"
    f["Displaced"] = float(prof.ailesinden_farkli_sehirde)
    tr["Displaced"] = "ogrenci.ailesinden_farkli_sehirde"
    f["Daytime/evening attendance"] = 1.0 if adm.ogretim_turu == "birinci" else 0.0
    tr["Daytime/evening attendance"] = "giris.ogretim_turu (birinci=1, ikinci=0)"
    f["Previous qualification"] = 1.0
    tr["Previous qualification"] = "sabit: lise/orta öğretim kodu (1)"
    m.assumptions.append("Önceki eğitim düzeyi 'lise mezunu' (UCI kodu 1) varsayıldı.")
    f["Application order"] = float(min(max(adm.tercih_sirasi - 1, 0), 9))
    tr["Application order"] = "giris.tercih_sirasi - 1 (0-9 aralığına kırpıldı)"

    # ---- mali (modelde karşılığı olanlar) ----
    has_burs = (
        fin.kyk_durumu in ("burs", "kredi_ve_burs")
        or fin.diger_burs
        or (fin.vakif_burs_orani > 0 and adm.universite_turu == "vakif")
    )
    f["Scholarship holder"] = float(has_burs)
    tr["Scholarship holder"] = "mali.kyk_durumu(burs) | mali.diger_burs | mali.vakif_burs_orani>0"
    if fin.vakif_burs_orani > 0 and adm.universite_turu != "vakif":
        m.warnings.append("vakif_burs_orani yalnızca vakıf üniversitesinde geçerlidir; devlet için yok sayıldı.")
    if fin.kyk_durumu == "kredi":
        m.assumptions.append("KYK kredisi geri ödemeli olduğu için 'burslu' sayılmadı.")

    tuition_applicable = adm.universite_turu == "vakif" or adm.ogretim_turu == "ikinci"
    if fin.harc_odemesi_guncel is None:
        tuition_ok = True
        if tuition_applicable:
            m.warnings.append(
                "Harçlı programda 'harc_odemesi_guncel' girilmedi; güncel (true) varsayıldı. "
                "Bu alan modelin en etkili mali göstergelerinden biridir."
            )
    else:
        tuition_ok = fin.harc_odemesi_guncel
    f["Tuition fees up to date"] = float(tuition_ok)
    tr["Tuition fees up to date"] = "mali.harc_odemesi_guncel (devlet+birinci öğretimde varsayılan 1)"
    f["Debtor"] = float(fin.borclu)
    tr["Debtor"] = "mali.borclu"

    if prof.ailesinden_farkli_sehirde and fin.barinma == "aile_yaninda":
        m.warnings.append("Tutarsızlık: öğrenci ailesinden farklı şehirde ama barinma='aile_yaninda'.")

    # ---- giriş puanları: kohort içi standardizasyon ----
    train_adm_mean = art.scaler_mean.get("Admission grade", 126.9)
    train_adm_std = art.scaler_scale.get("Admission grade", 14.5)
    train_prev_mean = art.scaler_mean.get("Previous qualification (grade)", 132.7)
    train_prev_std = art.scaler_scale.get("Previous qualification (grade)", 13.2)

    if adm.admission_grade_0_200 is not None:
        f["Admission grade"] = adm.admission_grade_0_200
        tr["Admission grade"] = "giris.admission_grade_0_200 (doğrudan)"
    elif adm.yks_yerlestirme_puani is not None and YKS_COHORT_MEAN is not None and YKS_COHORT_STD not in (None, 0):
        f["Admission grade"] = cohort_to_model_scale(
            adm.yks_yerlestirme_puani, YKS_COHORT_MEAN, YKS_COHORT_STD, train_adm_mean, train_adm_std
        )
        tr["Admission grade"] = "YKS puanı -> kohort z-skoru -> model ölçeği"
    else:
        if adm.yks_yerlestirme_puani is not None:
            m.warnings.append(
                "YKS_COHORT_MEAN/YKS_COHORT_STD tanımlı olmadığından YKS puanı modele "
                "katılmadı (eğitim ortalaması kullanıldı)."
            )
        m.assumptions.append("Giriş notu için eğitim ortalaması kullanıldı.")

    if adm.lise_diploma_notu is not None and LISE_COHORT_MEAN is not None and LISE_COHORT_STD not in (None, 0):
        f["Previous qualification (grade)"] = cohort_to_model_scale(
            adm.lise_diploma_notu, LISE_COHORT_MEAN, LISE_COHORT_STD, train_prev_mean, train_prev_std
        )
        tr["Previous qualification (grade)"] = "lise diploma notu -> kohort z-skoru -> model ölçeği"
    else:
        if adm.lise_diploma_notu is not None:
            m.warnings.append(
                "LISE_COHORT_MEAN/LISE_COHORT_STD tanımlı olmadığından diploma notu modele "
                "katılmadı (eğitim ortalaması kullanıldı)."
            )
        m.assumptions.append("Önceki eğitim notu için eğitim ortalaması kullanıldı.")

    # ---- akademik (iki yarıyıl) ----
    for key, sem, label in (("1st", p.akademik.donem_1, "donem_1"), ("2nd", p.akademik.donem_2, "donem_2")):
        base = f"Curricular units {key} sem"
        f[f"{base} (enrolled)"] = float(sem.alinan_ders)
        f[f"{base} (approved)"] = float(sem.gecilen_ders)
        f[f"{base} (credited)"] = float(sem.muaf_ders)
        f[f"{base} (without evaluations)"] = float(sem.devamsiz_ders)
        f[f"{base} (grade)"] = float(sem.grade_on_20_scale())
        if sem.sinav_sayisi is not None:
            f[f"{base} (evaluations)"] = float(sem.sinav_sayisi)
        else:
            ratio = art.scaler_mean.get(f"{base} (evaluations)", 8.0) / max(
                art.scaler_mean.get(f"{base} (enrolled)", 6.0), 1e-9
            )
            f[f"{base} (evaluations)"] = round(sem.alinan_ders * ratio)
            m.warnings.append(
                f"akademik.{label}.sinav_sayisi girilmedi; eğitim oranıyla " f"({ratio:.2f} sınav/ders) tahmin edildi."
            )
        for suffix, src in (
            ("enrolled", "alinan_ders"),
            ("approved", "gecilen_ders"),
            ("credited", "muaf_ders"),
            ("without evaluations", "devamsiz_ders"),
            ("grade", "agno | vize/final ağırlıklı ort."),
            ("evaluations", "sinav_sayisi"),
        ):
            tr[f"{base} ({suffix})"] = f"akademik.{label}.{src}"
        m.derived[f"grade20_{label}"] = f[f"{base} (grade)"]
        m.derived[f"pass_ratio_{label}"] = (sem.gecilen_ders / sem.alinan_ders) if sem.alinan_ders else None

    # ---- makroekonomi: Portekiz dönemine ait; Türkiye için nötr ----
    m.assumptions.append(
        "İşsizlik/enflasyon/GSYH eğitim ortalamasına sabitlendi: model Portekiz 2008-2019 "
        "makro verisiyle eğitildi, Türkiye değerleri (özellikle enflasyon) dağılım dışıdır."
    )

    # ---- ileri düzey: ham sütun ezme ----
    if p.raw_features:
        for name, value in p.raw_features.items():
            f[name] = value
            tr[name] = "raw_features (kullanıcı tarafından doğrudan verildi)"

    m.derived.update(
        has_burs=has_burs,
        tuition_applicable=tuition_applicable,
        tuition_ok=tuition_ok,
        universite_turu=adm.universite_turu,
    )
    return m


# =============================================================================
# 7. GÜVENLİ ÖN İŞLEME (inference-time pipeline)
# =============================================================================

_ONEHOT_RE = re.compile(r"^(.*)_(\d+)$")


def build_model_frame(art: Artifacts, features: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
    """Ham özellik sözlüğünü modelin beklediği (1 x N) ölçeklenmiş DataFrame'e çevirir.

    Garantiler:
      1. Çıktı sütunları BİREBİR `model_columns.pkl` sırasındadır (eksik one-hot
         sütunları 0 = get_dummies(drop_first=True) referans kategorisi).
      2. Eksik SAYISAL sütunlar eğitim ortalamasıyla doldurulur (ölçekli değer 0),
         0 ile değil. Ham 0 doldurmak, ölçekleyiciden geçince büyük ve anlamsız
         bir sapmaya yol açardı.
      3. Bilinmeyen sütun adları sessizce atılmaz; yakın eşleşme önerisiyle 422 döner.
      4. Değerler sonlu olmalı; one-hot sütunlar yalnızca 0/1 olabilir ve aynı
         grup içinde (örn. Course_*) birden fazla 1 olamaz.
      5. Scaler yalnızca eğitimde ölçeklenen sütunlara uygulanır; ikili ve one-hot
         sütunlar dokunulmadan kalır. Boyut/NaN/inf kontrolü yapılır.

    Returns:
        (ölçeklenmiş DataFrame, eğitim dağılımı dışı değerler için uyarı listesi)
    """
    columns, numeric = art.columns, art.numeric_cols
    column_set = set(columns)

    unknown = [k for k in features if k not in column_set]
    if unknown:
        hints = {k: difflib.get_close_matches(k, columns, n=3, cutoff=0.6) for k in unknown}
        raise PreprocessingError(
            "unknown_feature",
            "Modelde bulunmayan özellik adı(ları) gönderildi.",
            {"bilinmeyen": unknown, "belki_kastettiniz": hints},
        )

    row: dict[str, float] = {c: 0.0 for c in columns}
    row.update({c: art.scaler_mean[c] for c in numeric})
    for name, value in features.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise PreprocessingError(
                "invalid_feature_type",
                f"'{name}' için sayısal değer bekleniyor.",
                {"ozellik": name, "gelen_tip": type(value).__name__},
            ) from None
        if not math.isfinite(number):
            raise PreprocessingError("non_finite_feature", f"'{name}' sonlu bir sayı olmalı.", {"ozellik": name})
        row[name] = number

    # one-hot doğrulaması (yalnızca scaler dışı ve '_<sayı>' ile biten sütunlar)
    numeric_set = set(numeric)
    groups: dict[str, list[str]] = {}
    for name in features:
        match = _ONEHOT_RE.match(name)
        if name not in numeric_set and match:
            if row[name] not in (0.0, 1.0):
                raise PreprocessingError(
                    "invalid_onehot_value", f"'{name}' yalnızca 0 veya 1 olabilir.", {"ozellik": name}
                )
            if row[name] == 1.0:
                groups.setdefault(match.group(1), []).append(name)
    clash = {g: cols for g, cols in groups.items() if len(cols) > 1}
    if clash:
        raise PreprocessingError("multiple_categories", "Aynı kategori grubunda birden fazla değer 1 olamaz.", clash)

    frame = pd.DataFrame([row], columns=columns, dtype="float64")

    # dağılım dışı (OOD) kontrol - ölçekleme öncesi z-skoru
    ood: list[str] = []
    for c in numeric:
        z = (frame.at[0, c] - art.scaler_mean[c]) / art.scaler_scale[c]
        if abs(z) > OOD_Z_THRESHOLD:
            ood.append(f"'{c}' eğitim dağılımının dışında (z={z:+.1f}); tahmin güvenilirliği düşebilir.")

    try:
        scaled = art.scaler.transform(frame[numeric])
    except Exception as exc:  # noqa: BLE001
        raise PreprocessingError(
            "scaling_failed", "Ölçekleme aşamasında boyut/tip uyuşmazlığı.", {"hata": f"{type(exc).__name__}: {exc}"}
        ) from exc
    scaled = np.asarray(scaled, dtype="float64")
    if scaled.shape != (1, len(numeric)) or not np.isfinite(scaled).all():
        raise PreprocessingError("scaling_invalid", "Ölçekleme sonucu geçersiz (NaN/Inf veya yanlış boyut).")
    frame[numeric] = scaled

    if frame.shape != (1, len(columns)) or frame.isna().any().any():
        raise PreprocessingError(
            "frame_invalid",
            "Model girdisi oluşturulamadı (boyut veya eksik değer).",
            {"beklenen_sutun": len(columns), "gelen": int(frame.shape[1])},
        )
    return frame, ood


# =============================================================================
# 8. TAHMİN VE RİSK / DANIŞMAN KURALLARI
# =============================================================================


def predict_proba_safe(art: Artifacts, frame: pd.DataFrame) -> np.ndarray:
    """predict_proba çağrısını sarmalar; sonuçların geçerliliğini doğrular."""
    try:
        proba = np.asarray(art.model.predict_proba(frame), dtype=float)
    except Exception as exc:  # noqa: BLE001
        logger.exception("predict_proba başarısız")
        raise ApiError(500, "prediction_failed", "Model tahmini üretilemedi.", {"hata": type(exc).__name__}) from exc
    if proba.ndim != 2 or proba.shape[1] != 3 or not np.isfinite(proba).all():
        raise ApiError(500, "prediction_invalid", "Model geçersiz bir olasılık çıktısı üretti.")
    return proba


def risk_level_from(probs: dict[str, float]) -> str:
    """P(Dropout) ve en olası sınıfa göre 3 kademeli risk seviyesi."""
    top = max(probs, key=probs.get)  # type: ignore[arg-type]
    p_drop = probs["Dropout"]
    if top == "Dropout" or p_drop >= HIGH_RISK_THRESHOLD:
        return "yuksek"
    if p_drop >= MEDIUM_RISK_THRESHOLD:
        return "orta"
    return "dusuk"


def build_advisories(p: StudentPayload, m: MappedInput) -> list[dict[str, str]]:
    """Modelde OLMAYAN Türkiye özelliklerinden şeffaf, kural tabanlı uyarılar üretir.

    Bu uyarılar olasılıkları değiştirmez. Eşikler mutlak TL cinsinden değil oranla
    tanımlanır; çünkü enflasyon nedeniyle sabit TL eşikleri hızla geçersizleşir.
    """
    out: list[dict[str, str]] = []
    fin, adm = p.mali, p.giris

    def add(code: str, severity: str, text: str) -> None:
        out.append({"kod": code, "seviye": severity, "mesaj": text})

    # --- aylık bütçe dengesi ---
    balance, ratio = fin.aylik_butce_dengesi_try, None
    if fin.aylik_gelir_try is not None and fin.aylik_gider_try is not None:
        balance = fin.aylik_gelir_try - fin.aylik_gider_try
        ratio = fin.aylik_gelir_try / fin.aylik_gider_try if fin.aylik_gider_try > 0 else None
    if balance is not None and balance < 0:
        severe = ratio is not None and ratio < 0.70
        add(
            "butce_acigi",
            "yuksek" if severe else "orta",
            "Aylık bütçe açığı var"
            + (f" (gelir/gider = {ratio:.2f})" if ratio is not None else "")
            + ". Acil burs/yardım fonları ve barınma desteği değerlendirilmeli.",
        )
    if balance is not None and balance < 0 and fin.barinma == "kirada":
        add(
            "kira_yuku",
            "yuksek",
            "Kirada barınan ve bütçe açığı olan öğrenci: barınma maliyeti terk riskini artıran "
            "bir stres faktörüdür; KYK yurdu/öğrenci evi alternatifleri görüşülmeli.",
        )
    if fin.kyk_durumu == "kredi" and balance is not None and balance < 0:
        add(
            "kredi_butce",
            "orta",
            "Yalnızca KYK kredisi alıyor ve bütçe açığı var: geri ödemesiz burs " "seçenekleri araştırılmalı.",
        )
    if p.ogrenci.ailesinden_farkli_sehirde and fin.barinma in ("ozel_yurt", "kirada") and fin.kyk_durumu == "yok":
        add(
            "barinma_destegi",
            "orta",
            "Ailesinden ayrı şehirde, KYK desteği olmadan barınıyor: KYK yurt/burs " "başvuru durumu kontrol edilmeli.",
        )

    # --- kısmi zamanlı çalışma ---
    if fin.kismi_zamanli_calisiyor:
        hours = fin.haftalik_calisma_saati
        if hours is None:
            add("calisma_belirsiz", "bilgi", "Kısmi zamanlı çalışıyor ancak haftalık saat bilgisi yok.")
        elif hours >= 20:
            add(
                "yogun_calisma",
                "yuksek",
                f"Haftada {hours:.0f} saat çalışıyor: ders/sınav yüküyle çakışma riski yüksek; "
                "esnek devam ve sınav programı seçenekleri görüşülmeli.",
            )
        elif hours >= 10:
            add("orta_calisma", "orta", f"Haftada {hours:.0f} saat çalışıyor: akademik takip önerilir.")

    # --- bölüm memnuniyeti ---
    sat = p.anket.bolum_memnuniyeti
    if sat is not None and sat <= 2:
        add(
            "bolum_memnuniyetsizligi",
            "yuksek" if sat == 1 else "orta",
            "Bölümden memnuniyet düşük: kariyer danışmanlığı, çift anadal/yatay geçiş ve "
            "bölüm içi uyum çalışmaları değerlendirilebilir.",
        )
    if sat is not None and sat <= 2 and adm.tercih_sirasi > 3:
        add(
            "istemsiz_yerlesme",
            "orta",
            f"Bölüm {adm.tercih_sirasi}. tercihle kazanılmış ve memnuniyet düşük: "
            "yeniden sınava girme (tekrar YKS) eğilimi izlenmeli.",
        )

    # --- vakıf / harç ---
    if m.derived["tuition_applicable"] and not m.derived["tuition_ok"]:
        add(
            "harc_gecikmesi",
            "yuksek",
            "Harç/öğrenim ücreti güncel değil: vakıf ve ikinci öğretimde "
            "kayıt dondurma/silme riski doğurur; taksit/burs görüşmesi yapılmalı.",
        )
    if fin.borclu:
        add("borc", "yuksek", "Vadesi geçmiş okul/yurt borcu var.")

    # --- burs sürekliliği ---
    g_vals = [v for k, v in m.derived.items() if k.startswith("grade20_")]
    if m.derived["has_burs"] and g_vals and min(g_vals) < 10:
        add(
            "burs_devam_riski",
            "orta",
            "Burslu öğrencinin dönem ortalaması düşük: burs/kredi devam koşulları "
            "(not ortalaması, azami öğrenim süresi) ilgili yönetmelikten kontrol edilmeli.",
        )
    return out


# =============================================================================
# 9. ÖZELLİK ÖNEMİ - TÜRKİYE BAĞLAMI METASI
# =============================================================================
# uyum: dogrudan = Türkiye verisinden birebir doldurulabilir
#       yaklasik = dönüşümle doldurulur, dağılım farkı olabilir
#       uyumsuz  = Türkiye karşılığı yok; sabitlenir (bu özelliklerin önemi
#                  Türkiye için bilgi taşımaz, yalnızca Portekiz örüntüsünü yansıtır)
_META: dict[str, dict[str, str]] = {
    "Curricular units 2nd sem (approved)": {
        "ad": "2. dönem geçilen ders sayısı",
        "uyum": "dogrudan",
        "alan": "akademik.donem_2.gecilen_ders",
        "baglam": (
            "Vize-final sonucu kalınan dersler (FF/FD, devamsızlık) doğrudan düşer; Türkiye'de de ilk erken "
            "uyarı sinyalidir."
        ),
    },
    "Curricular units 2nd sem (grade)": {
        "ad": "2. dönem not ortalaması",
        "uyum": "yaklasik",
        "alan": "akademik.donem_2.agno | vize/final",
        "baglam": (
            "AGNO x5 ile 0-20'ye çevrilir. Vize ağırlığı (%30-50) kurumdan kuruma değiştiği için "
            "'vize_agirligi' ayarlanabilir."
        ),
    },
    "Curricular units 1st sem (approved)": {
        "ad": "1. dönem geçilen ders sayısı",
        "uyum": "dogrudan",
        "alan": "akademik.donem_1.gecilen_ders",
        "baglam": "İlk yıl uyum sorunu (hazırlık, yabancı dil, lise-üniversite geçişi) burada görünür.",
    },
    "Curricular units 1st sem (grade)": {
        "ad": "1. dönem not ortalaması",
        "uyum": "yaklasik",
        "alan": "akademik.donem_1.agno | vize/final",
        "baglam": "Vize-final ağırlıklı not sisteminin 0-20 ölçeğine çevrilmiş hali.",
    },
    "Curricular units 2nd sem (evaluations)": {
        "ad": "2. dönem girilen sınav sayısı",
        "uyum": "yaklasik",
        "alan": "akademik.donem_2.sinav_sayisi",
        "baglam": "Vize+final+bütünleme sayısı; sınava girmemek (devamsız/girmedi) terk habercisidir.",
    },
    "Curricular units 1st sem (evaluations)": {
        "ad": "1. dönem girilen sınav sayısı",
        "uyum": "yaklasik",
        "alan": "akademik.donem_1.sinav_sayisi",
        "baglam": "Vize+final+bütünleme sayısı.",
    },
    "Curricular units 2nd sem (enrolled)": {
        "ad": "2. dönem alınan ders",
        "uyum": "dogrudan",
        "alan": "akademik.donem_2.alinan_ders",
        "baglam": "Ders yükü; alttan ders yükü ve kredi sınırı Türkiye'de sık görülür.",
    },
    "Curricular units 1st sem (enrolled)": {
        "ad": "1. dönem alınan ders",
        "uyum": "dogrudan",
        "alan": "akademik.donem_1.alinan_ders",
        "baglam": "Ders yükü.",
    },
    "Curricular units 1st sem (credited)": {
        "ad": "1. dönem muaf ders",
        "uyum": "dogrudan",
        "alan": "akademik.donem_1.muaf_ders",
        "baglam": "Yatay/dikey geçiş, DGS, intibak muafiyetleri.",
    },
    "Curricular units 2nd sem (credited)": {
        "ad": "2. dönem muaf ders",
        "uyum": "dogrudan",
        "alan": "akademik.donem_2.muaf_ders",
        "baglam": "Yatay/dikey geçiş, DGS, intibak muafiyetleri.",
    },
    "Curricular units 1st sem (without evaluations)": {
        "ad": "1. dönem değerlendirmesiz ders",
        "uyum": "yaklasik",
        "alan": "akademik.donem_1.devamsiz_ders",
        "baglam": "Devamsızlıktan kalma (DZ) ve sınava girmeme.",
    },
    "Curricular units 2nd sem (without evaluations)": {
        "ad": "2. dönem değerlendirmesiz ders",
        "uyum": "yaklasik",
        "alan": "akademik.donem_2.devamsiz_ders",
        "baglam": "Devamsızlıktan kalma (DZ) ve sınava girmeme.",
    },
    "Age at enrollment": {
        "ad": "Kayıt yaşı",
        "uyum": "dogrudan",
        "alan": "ogrenci.kayit_yasi",
        "baglam": (
            "Türkiye'de tekrar YKS'ye girip yerleşen, çalışırken okuyan veya DGS/ikinci üniversite "
            "öğrencileri daha yüksek yaşta kayıt olur."
        ),
    },
    "Tuition fees up to date": {
        "ad": "Harç/öğrenim ücreti güncel mi",
        "uyum": "yaklasik",
        "alan": "mali.harc_odemesi_guncel",
        "baglam": (
            "Devlet birinci öğretimde harç yoktur; asıl bilgi vakıf üniversitelerinde ve ikinci öğretimde "
            "ortaya çıkar."
        ),
    },
    "Admission grade": {
        "ad": "Giriş notu (YKS yerleştirme puanı)",
        "uyum": "yaklasik",
        "alan": "giris.yks_yerlestirme_puani",
        "baglam": "Kohort içi standardizasyon gerekir; aynı puan devlet/vakıf ve bölüme göre farklı anlam taşır.",
    },
    "Previous qualification (grade)": {
        "ad": "Önceki eğitim notu (lise diploma notu)",
        "uyum": "yaklasik",
        "alan": "giris.lise_diploma_notu",
        "baglam": "Lise not enflasyonu nedeniyle kohort içi standardizasyon önerilir.",
    },
    "Scholarship holder": {
        "ad": "Burs durumu",
        "uyum": "yaklasik",
        "alan": "mali.kyk_durumu | diger_burs | vakif_burs_orani",
        "baglam": (
            "KYK bursu, vakıf üniversitesi %50/%100 bursu ve kurum bursları; KYK kredisi (geri ödemeli) "
            "burs sayılmaz."
        ),
    },
    "Debtor": {
        "ad": "Borçlu mu",
        "uyum": "yaklasik",
        "alan": "mali.borclu",
        "baglam": "Vadesi geçmiş okul/yurt borcu; KYK kredi borcu mezuniyet sonrası başladığı için sayılmaz.",
    },
    "Displaced": {
        "ad": "Ailesinden farklı şehirde",
        "uyum": "yaklasik",
        "alan": "ogrenci.ailesinden_farkli_sehirde",
        "baglam": "Barınma maliyeti ve KYK yurt kontenjanı; İstanbul gibi metropollerde kira yükü belirleyicidir.",
    },
    "Application order": {
        "ad": "Tercih sırası",
        "uyum": "yaklasik",
        "alan": "giris.tercih_sirasi",
        "baglam": "İstemsiz yerleşme (düşük tercih) bölüm memnuniyetsizliği ve tekrar YKS eğilimiyle ilişkilidir.",
    },
    "Gender": {
        "ad": "Cinsiyet",
        "uyum": "dogrudan",
        "alan": "ogrenci.cinsiyet",
        "baglam": (
            "Adalet (fairness) riski: kararlar cinsiyete göre farklılaşmamalı; kurum içi etik incelemesi " "önerilir."
        ),
    },
    "Daytime/evening attendance": {
        "ad": "Öğretim türü (gündüz/akşam)",
        "uyum": "dogrudan",
        "alan": "giris.ogretim_turu",
        "baglam": "Birinci/ikinci öğretim ayrımı; ikinci öğretim harçlıdır ve öğrenci profili farklıdır.",
    },
    "International": {
        "ad": "Yabancı uyruklu",
        "uyum": "dogrudan",
        "alan": "ogrenci.uyruk",
        "baglam": "Türkiye'de yabancı uyruklu öğrenci profili (dil, vize, burs) Portekiz'den farklıdır.",
    },
    "Educational special needs": {
        "ad": "Özel eğitim ihtiyacı",
        "uyum": "dogrudan",
        "alan": "ogrenci.ozel_egitim_ihtiyaci",
        "baglam": "Engelli öğrenci birimi desteği.",
    },
    "Previous qualification": {
        "ad": "Önceki eğitim türü",
        "uyum": "yaklasik",
        "alan": "(sabit: lise)",
        "baglam": "Meslek lisesi/açık öğretim/ikinci üniversite ayrımı eşlenmez.",
    },
    "Unemployment rate": {
        "ad": "İşsizlik oranı",
        "uyum": "uyumsuz",
        "alan": "-",
        "baglam": (
            "Portekiz 2008-2019 verisi; Türkiye için sabitlenir. Önemi kohort/yıl etkisini (zaman vekili) "
            "yansıtıyor olabilir."
        ),
    },
    "Inflation rate": {
        "ad": "Enflasyon oranı",
        "uyum": "uyumsuz",
        "alan": "-",
        "baglam": "Portekiz'de ~%-1..4; Türkiye enflasyonu dağılım dışıdır, modele girmez.",
    },
    "GDP": {
        "ad": "GSYH büyümesi",
        "uyum": "uyumsuz",
        "alan": "-",
        "baglam": "Portekiz makro verisi; Türkiye için sabitlenir.",
    },
    "Course": {
        "ad": "Bölüm/program (Portekiz kodları)",
        "uyum": "uyumsuz",
        "alan": "raw_features",
        "baglam": (
            "Portekiz programları; Türkiye bölümleri eşlenemez. Türkiye verisiyle yeniden eğitimde en "
            "kritik eklenmesi gereken özelliktir."
        ),
    },
    "Application mode": {
        "ad": "Başvuru yolu (Portekiz)",
        "uyum": "uyumsuz",
        "alan": "raw_features",
        "baglam": "Türkiye karşılığı: YKS genel / yatay geçiş / DGS / af / şehit-gazi kontenjanı; eşlenmez.",
    },
    "Mother's occupation": {
        "ad": "Anne mesleği (Portekiz kodları)",
        "uyum": "uyumsuz",
        "alan": "raw_features",
        "baglam": "Sosyoekonomik göstergedir, ancak kodlar Portekiz'e özgüdür.",
    },
    "Father's occupation": {
        "ad": "Baba mesleği (Portekiz kodları)",
        "uyum": "uyumsuz",
        "alan": "raw_features",
        "baglam": "Sosyoekonomik göstergedir, ancak kodlar Portekiz'e özgüdür.",
    },
    "Mother's qualification": {
        "ad": "Anne eğitim düzeyi (Portekiz kodları)",
        "uyum": "uyumsuz",
        "alan": "raw_features",
        "baglam": "Birinci kuşak üniversiteli olma Türkiye'de de güçlü bir risk etkenidir; yeniden eğitimde eklenmeli.",
    },
    "Father's qualification": {
        "ad": "Baba eğitim düzeyi (Portekiz kodları)",
        "uyum": "uyumsuz",
        "alan": "raw_features",
        "baglam": "Birinci kuşak üniversiteli olma Türkiye'de de güçlü bir risk etkenidir; yeniden eğitimde eklenmeli.",
    },
    "Marital status": {
        "ad": "Medeni durum",
        "uyum": "uyumsuz",
        "alan": "raw_features",
        "baglam": "Referans kategori (bekar) varsayılır.",
    },
    "Nacionality": {
        "ad": "Uyruk (Portekiz kodları)",
        "uyum": "uyumsuz",
        "alan": "raw_features",
        "baglam": "Referans kategoriye sabitlenir.",
    },
}

_MODEL_DISI = [
    {
        "alan": "mali.kyk_durumu / mali.barinma",
        "durum": (
            "Modelde sütunu yok; yalnızca 'Scholarship holder' ve 'Displaced' eşlemesine dolaylı katkı + "
            "danışman uyarısı."
        ),
    },
    {
        "alan": "mali.aylik_gelir_try / aylik_gider_try / aylik_butce_dengesi_try",
        "durum": "Yalnızca danışman uyarısı; olasılığı değiştirmez.",
    },
    {"alan": "anket.bolum_memnuniyeti", "durum": "Yalnızca danışman uyarısı; olasılığı değiştirmez."},
    {
        "alan": "mali.kismi_zamanli_calisiyor / haftalik_calisma_saati",
        "durum": "Yalnızca danışman uyarısı; olasılığı değiştirmez.",
    },
]


def _group_of(column: str) -> str:
    """'Course_9500' -> 'Course'; sayısal/ikili sütunlar kendi adıdır."""
    if column in _META:
        return column
    match = _ONEHOT_RE.match(column)
    return match.group(1) if match else column


def _meta_for(group: str) -> dict[str, str]:
    return _META.get(group, {"ad": group, "uyum": "bilinmiyor", "alan": "-", "baglam": "Tanımlı Türkiye bağlamı yok."})


def compute_importance_report(art: Artifacts, top_k: int, grouped: bool) -> dict[str, Any]:
    """MDI özellik önemini (isteğe bağlı gruplanmış) Türkiye bağlamıyla raporlar."""
    imp = art.importances
    total = float(imp.sum()) or 1.0
    cols = art.columns
    group_names = [_group_of(c) for c in cols]

    if grouped:
        frame = pd.DataFrame({"key": group_names, "imp": imp.values})
        agg = frame.groupby("key")["imp"].sum().sort_values(ascending=False)
        # grup std: her ağaçta grup toplamının ağaçlar arası standart sapması
        idx_by_group: dict[str, list[int]] = {}
        for i, g in enumerate(group_names):
            idx_by_group.setdefault(g, []).append(i)
        stds = {g: float(np.std(art.tree_importances[:, ix].sum(axis=1))) for g, ix in idx_by_group.items()}
        n_cols = {g: len(ix) for g, ix in idx_by_group.items()}
        ranked = [(g, float(v), stds[g], n_cols[g]) for g, v in agg.items()]
    else:
        stds_arr = art.tree_importances.std(axis=0)
        order = np.argsort(-imp.values)
        ranked = [(cols[i], float(imp.values[i]), float(stds_arr[i]), 1) for i in order]

    cumulative, items = 0.0, []
    for rank, (key, value, std, n) in enumerate(ranked, start=1):
        cumulative += value
        if rank > top_k:
            continue
        group = key if grouped else _group_of(key)
        meta = _meta_for(group)
        items.append(
            {
                "sira": rank,
                "ozellik": key,
                "onem": round(value, 5),
                "onem_yuzde": round(100 * value / total, 2),
                "kumulatif_yuzde": round(100 * cumulative / total, 2),
                "ağaçlar_arası_std": round(std, 5),
                "sutun_sayisi": n,
                "turkce_ad": meta["ad"],
                "turkiye_uyumu": meta["uyum"],
                "payload_alani": meta["alan"],
                "turkiye_baglami": meta["baglam"],
            }
        )

    share: dict[str, float] = {"dogrudan": 0.0, "yaklasik": 0.0, "uyumsuz": 0.0, "bilinmiyor": 0.0}
    for col, group in zip(cols, group_names):
        share[_meta_for(group)["uyum"]] += float(imp[col])
    share_pct = {k: round(100 * v / total, 2) for k, v in share.items()}

    return {
        "yontem": "MDI (Gini impurity decrease), ağaçlar üzerinden ortalama",
        "gruplanmis": grouped,
        "n_agac": art.model_stats["n_estimators"],
        "en_onemli": items,
        "turkiye_uyum_ozeti_yuzde": share_pct,
        "yorum": [
            f"Önemin %{share_pct['uyumsuz']:.1f}'i Türkiye'de doldurulamayan (Portekiz kodlu/makro) özelliklerde; "
            "bu pay Türkiye için bilgi taşımaz, sabit referansla çalışır.",
            f"Önemin %{share_pct['dogrudan'] + share_pct['yaklasik']:.1f}'i Türkiye verisinden "
            "doldurulabilen özelliklerde; "
            "bunların başında 1. ve 2. dönem geçilen ders sayısı ve not ortalaması gelir (vize-final sonucu).",
            "Yüksek önem = nedensellik değildir. Burs/harç gibi mali değişkenler ile terk arasındaki ilişki "
            "yönlü değil, ilişkisel bir örüntüdür.",
        ],
        "model_disi_turkiye_ozellikleri": _MODEL_DISI,
        "uyarilar": [
            "MDI önemi sürekli ve çok değerli değişkenleri kayırır; korelasyonlu değişkenlerin (1. ve 2. dönem) "
            "önemini böler. Kesin sıralama için ayrı bir doğrulama kümesinde permutation importance veya "
            "SHAP kullanın.",
            "Bu önem değerleri eğitim verisinden türetilmiştir; Türkiye'deki gerçek önem sırası yeniden "
            "eğitimle belirlenmelidir.",
        ],
    }


# =============================================================================
# 10. YANIT OLUŞTURMA
# =============================================================================


def _result_for(
    art: Artifacts,
    proba_row: np.ndarray,
    p: StudentPayload,
    m: MappedInput,
    ood: list[str],
    request_id: str,
    debug: bool,
) -> dict[str, Any]:
    probs = {name: float(v) for name, v in zip(art.class_names, proba_row)}
    top = max(probs, key=probs.get)  # type: ignore[arg-type]
    level = risk_level_from(probs)
    advisories = build_advisories(p, m)
    high_flags = [a for a in advisories if a["seviye"] == "yuksek"]
    manual_review = level != "dusuk" or bool(high_flags)

    result: dict[str, Any] = {
        "request_id": request_id,
        "ogrenci_ref": p.ogrenci_ref,
        "tahmin": {
            "sinif": top,
            "sinif_tr": CLASS_LABELS_TR[top],
            "olasiliklar": {k: round(v, 4) for k, v in probs.items()},
            "risk_seviyesi": level,
            "risk_seviyesi_tr": RISK_LABELS_TR[level],
            "esikler": {"yuksek": HIGH_RISK_THRESHOLD, "orta": MEDIUM_RISK_THRESHOLD},
        },
        "danisman_uyarilari": {
            "kullanim": "Model dışı: olasılıkları değiştirmez, yalnızca danışmana yön verir.",
            "uyarilar": advisories,
            "manuel_inceleme_onerilir": manual_review,
        },
        "uyarilar": m.warnings + ood,
        "varsayimlar": m.assumptions,
        "sorumluluk_notu": DISCLAIMER_TR,
    }
    if debug:
        result["debug"] = {"model_girdileri_ham": m.features, "eslesme_izi": m.trace}
    return result


# =============================================================================
# 11. UYGULAMA, ARA KATMAN VE HATA YAKALAYICILAR
# =============================================================================


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.load()  # başarısız olsa bile istisna fırlatmaz; servis 'degraded' ayağa kalkar
    yield


app = FastAPI(
    title="Öğrenci Terk Riski Erken Uyarı API (Türkiye)",
    version="2.0.0",
    description="Random Forest tabanlı erken uyarı servisi. UCI/Portekiz verisiyle eğitilmiş modelin "
    "Türkiye profiline uyarlanmış, doğrulamalı arayüzüdür. Ayrıntılar için /model-info.",
    lifespan=lifespan,
)


def _error_body(request: Request, code: str, message: str, details: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "hata": {"kod": code, "mesaj": message, "request_id": getattr(request.state, "request_id", None)}
    }
    if details is not None:
        body["hata"]["detay"] = details
    return body


@app.middleware("http")
async def request_context(request: Request, call_next):
    """İstek kimliği + süre logu + son çare hata yakalama. Payload ASLA loglanmaz (KVKK)."""
    request.state.request_id = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - hiçbir istisna süreci düşürmemeli
        logger.exception("Yakalanmamış hata rid=%s %s %s", request.state.request_id, request.method, request.url.path)
        response = JSONResponse(
            status_code=500,
            content=_error_body(
                request, "internal_error", "Beklenmeyen bir sunucu hatası oluştu. request_id ile destek isteyin."
            ),
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request.state.request_id
    logger.info(
        "rid=%s %s %s -> %d (%.1f ms)",
        request.state.request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status_code, content=_error_body(request, exc.code, exc.message, exc.details))


_MSG_TR = {
    "missing": "Zorunlu alan eksik",
    "extra_forbidden": "Tanımsız alan (yazım hatası olabilir)",
    "json_invalid": "Geçersiz JSON gövdesi",
    "model_attributes_type": "Nesne (JSON object) bekleniyor",
    "dict_type": "Nesne (JSON object) bekleniyor",
    "list_type": "Liste (JSON array) bekleniyor",
    "string_type": "Metin bekleniyor",
    "string_too_long": "Metin çok uzun",
    "too_short": "Liste boş olamaz",
    "bool_type": "Mantıksal değer bekleniyor",
}


def _translate_validation(err: dict[str, Any]) -> dict[str, Any]:
    etype, ctx = err.get("type", ""), err.get("ctx") or {}
    if etype in _MSG_TR:
        msg = _MSG_TR[etype]
    elif etype == "literal_error":
        msg = "Geçersiz değer. Kabul edilenler: " + str(ctx.get("expected", "?")).replace(" or ", " veya ")
    elif etype == "greater_than_equal":
        msg = f"Değer en az {ctx.get('ge')} olmalı"
    elif etype == "less_than_equal":
        msg = f"Değer en çok {ctx.get('le')} olmalı"
    elif etype == "value_error":
        msg = str(err.get("msg", "")).replace("Value error, ", "")
    else:
        msg = str(err.get("msg", "Geçersiz değer"))
    path = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
    if etype == "json_invalid":
        path = ""  # loc burada bayt konumudur, alan adı değil
    return {"alan": path or "(gövde)", "mesaj": msg, "tip": etype}


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    problems = [_translate_validation(e) for e in exc.errors()]
    return JSONResponse(
        status_code=422,
        content=_error_body(
            request,
            "validation_error",
            "İstek gövdesi doğrulanamadı (eksik alan, yanlış tip veya aralık dışı değer).",
            problems,
        ),
    )


@app.exception_handler(StarletteHTTPException)
async def http_handler(request: Request, exc: StarletteHTTPException):
    mapping = {
        404: "Endpoint bulunamadı.",
        405: "Bu endpoint için HTTP metodu geçersiz.",
        413: "İstek gövdesi çok büyük.",
    }
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(request, f"http_{exc.status_code}", mapping.get(exc.status_code, str(exc.detail))),
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logger.exception("İşlenmeyen hata rid=%s", getattr(request.state, "request_id", "-"))
    return JSONResponse(
        status_code=500, content=_error_body(request, "internal_error", "Beklenmeyen bir sunucu hatası oluştu.")
    )


# =============================================================================
# 12. ENDPOINT'LER (senkron `def`: FastAPI bunları iş parçacığı havuzunda çalıştırır,
#     böylece CPU-ağırlıklı sklearn çağrıları olay döngüsünü bloklamaz)
# =============================================================================


@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {"servis": app.title, "surum": app.version, "dokuman": "/docs", "durum": "/health"}


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness: süreç ayakta mı? Model yüklü olmasa da 200 döner, durumu bildirir."""
    err = store.error
    return {
        "durum": "saglikli" if store.ready else "degraded",
        "model_hazir": store.ready,
        "hata": None if err is None else {"kod": err.code, "mesaj": err.message},
    }


@app.get("/ready")
def ready() -> JSONResponse:
    """Readiness: trafik almaya hazır mı? Model yoksa 503 (yük dengeleyici dışlasın)."""
    try:
        art = store.get()
    except ApiError as exc:
        return JSONResponse(status_code=503, content={"hazir": False, "neden": exc.message, "detay": exc.details})
    return JSONResponse(status_code=200, content={"hazir": True, "sutun_sayisi": len(art.columns)})


@app.get("/template")
def template() -> dict[str, Any]:
    """Örnek payload ve kabul edilen değerler (istemci geliştiricileri için)."""
    return {
        "aciklama": "Zorunlu: ogrenci.kayit_yasi, ogrenci.cinsiyet, akademik.donem_1, akademik.donem_2. "
        'Diğer her şey isteğe bağlıdır. Ondalık için 12.5 veya "12,5" kabul edilir.',
        "ornek": EXAMPLE_PAYLOAD,
        "kabul_edilen_degerler": {
            "cinsiyet": ["erkek", "kadin"],
            "uyruk": ["tc", "yabanci"],
            "universite_turu": ["devlet", "vakif"],
            "ogretim_turu": ["birinci", "ikinci"],
            "kyk_durumu": ["yok", "kredi", "burs", "kredi_ve_burs"],
            "barinma": ["aile_yaninda", "kyk_yurdu", "ozel_yurt", "kirada"],
            "mantiksal": "true/false, 1/0, evet/hayır",
        },
        "model_disi_alanlar": _MODEL_DISI,
        "json_schema_ayrintisi": "/openapi.json",
    }


EXAMPLE_PAYLOAD: dict[str, Any] = {
    "ogrenci_ref": "ogr-000123",
    "ogrenci": {
        "kayit_yasi": 19,
        "cinsiyet": "kadin",
        "uyruk": "tc",
        "ozel_egitim_ihtiyaci": False,
        "ailesinden_farkli_sehirde": True,
    },
    "giris": {
        "universite_turu": "vakif",
        "ogretim_turu": "birinci",
        "yks_yerlestirme_puani": 372.5,
        "lise_diploma_notu": 84.2,
        "tercih_sirasi": 4,
    },
    "akademik": {
        "donem_1": {
            "alinan_ders": 6,
            "gecilen_ders": 4,
            "sinav_sayisi": 10,
            "muaf_ders": 0,
            "devamsiz_ders": 1,
            "vize_ortalamasi": 52,
            "final_ortalamasi": 48,
            "vize_agirligi": 0.4,
        },
        "donem_2": {
            "alinan_ders": 6,
            "gecilen_ders": 3,
            "sinav_sayisi": 9,
            "muaf_ders": 0,
            "devamsiz_ders": 2,
            "agno": 1.8,
        },
    },
    "mali": {
        "kyk_durumu": "kredi",
        "barinma": "kirada",
        "diger_burs": False,
        "vakif_burs_orani": 0,
        "harc_odemesi_guncel": False,
        "borclu": False,
        "aylik_gelir_try": 12000,
        "aylik_gider_try": 21000,
        "kismi_zamanli_calisiyor": True,
        "haftalik_calisma_saati": 24,
    },
    "anket": {"bolum_memnuniyeti": 2},
}


@app.post("/predict")
def predict(
    payload: StudentPayload,
    request: Request,
    debug: bool = Query(False, description="Model girdilerini ve eşleme izini de döndür"),
) -> dict[str, Any]:
    """Tek öğrenci için terk riski tahmini.

    Hata kodları: 422 (doğrulama/ön işleme), 503 (model yok/uyumsuz), 500 (beklenmeyen).
    """
    art = store.get()
    mapped = map_payload_to_features(art, payload)
    frame, ood = build_model_frame(art, mapped.features)
    proba = predict_proba_safe(art, frame)[0]
    return _result_for(art, proba, payload, mapped, ood, request.state.request_id, debug)


@app.post("/predict/batch")
def predict_batch(body: BatchRequest, request: Request) -> dict[str, Any]:
    """Çoklu tahmin. Bozuk bir kayıt yalnızca kendi satırında hata üretir; diğerleri işlenir."""
    art = store.get()
    rid = request.state.request_id
    items: list[dict[str, Any]] = [{} for _ in body.ogrenciler]
    ok_idx: list[int] = []
    frames: list[pd.DataFrame] = []
    context: dict[int, tuple[StudentPayload, MappedInput, list[str]]] = {}

    for i, raw in enumerate(body.ogrenciler):
        try:
            payload = StudentPayload.model_validate(raw)
            mapped = map_payload_to_features(art, payload)
            frame, ood = build_model_frame(art, mapped.features)
        except Exception as exc:  # noqa: BLE001 - kayıt bazında izolasyon
            if hasattr(exc, "errors") and callable(exc.errors):  # pydantic.ValidationError
                detail: Any = [_translate_validation(e) for e in exc.errors()]
                code, msg = "validation_error", "Kayıt doğrulanamadı."
            elif isinstance(exc, ApiError):
                code, msg, detail = exc.code, exc.message, exc.details
            else:
                logger.exception("Batch kaydı işlenemedi rid=%s index=%d", rid, i)
                code, msg, detail = "internal_error", "Kayıt işlenirken beklenmeyen hata.", None
            items[i] = {"index": i, "basarili": False, "hata": {"kod": code, "mesaj": msg, "detay": detail}}
            continue
        ok_idx.append(i)
        frames.append(frame)
        context[i] = (payload, mapped, ood)

    if frames:
        probas = predict_proba_safe(art, pd.concat(frames, ignore_index=True))
        for row, i in zip(probas, ok_idx):
            payload, mapped, ood = context[i]
            items[i] = {"index": i, "basarili": True, "sonuc": _result_for(art, row, payload, mapped, ood, rid, False)}

    n_ok = len(ok_idx)
    return {"request_id": rid, "toplam": len(items), "basarili": n_ok, "hatali": len(items) - n_ok, "sonuclar": items}


@app.get("/feature-importance")
def feature_importance(
    top_k: int = Query(15, ge=1, le=300, description="Döndürülecek en önemli özellik/grup sayısı"),
    grouped: bool = Query(
        True, description="One-hot sütunları (Course_*, Father's occupation_* ...) grup olarak topla"
    ),
) -> dict[str, Any]:
    """Random Forest özellik önemi + Türkiye bağlamı + 'bu önemin ne kadarı Türkiye'de doldurulabilir' özeti."""
    art = store.get()
    try:
        return compute_importance_report(art, top_k, grouped)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Özellik önemi raporu üretilemedi")
        raise ApiError(
            500, "importance_failed", "Özellik önemi raporu üretilemedi.", {"hata": type(exc).__name__}
        ) from exc


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Model künyesi, sürüm uyarıları ve aşırı öğrenme teşhisi."""
    art = store.get()
    s = art.model_stats
    overfit_risk = "yuksek" if (s["max_depth_param"] is None and s["min_samples_leaf_param"] == 1) else "orta/dusuk"
    return {
        "model": type(art.model).__name__,
        "egitim_verisi": "UCI Predict Students' Dropout and Academic Success (Portekiz, 2008-2019)",
        "turkiye_icin_kalibre_mi": False,
        "sinif_eslemesi": dict(
            zip([int(c) if isinstance(c, (int, np.integer)) else str(c) for c in art.model.classes_], art.class_names)
        ),
        "sutun_sayisi": len(art.columns),
        "olceklenen_sayisal_sutun_sayisi": len(art.numeric_cols),
        "agac_istatistikleri": s,
        "asiri_ogrenme_teshisi": {
            "risk": overfit_risk,
            "gerekce": (
                "min_samples_leaf=1 ve max_depth=None: ağaçlar yaprak başına tek örneğe kadar büyür "
                "(ezberleme eğilimi). Eğitim doğruluğu yüksek görünse de genelleme daha düşük olabilir."
                if overfit_risk == "yuksek"
                else "Ağaç kısıtları tanımlı."
            ),
            "oneriler": [
                "min_samples_leaf 5-20 ve max_depth 8-16 ile yeniden eğitim",
                "Zaman bazlı doğrulama ve stratified K-Fold; eğitim/doğrulama farkını izleyin",
                "Dropout sınıfı için recall/F1 ve Brier skoru; CalibratedClassifierCV ile kalibrasyon",
                "Permutation importance / SHAP'i ayrı doğrulama kümesinde hesaplayın",
            ],
        },
        "yukleme_uyarilari": art.load_warnings,
        "yuklenme_zamani_epoch": int(art.loaded_at),
        "risk_esikleri": {"yuksek": HIGH_RISK_THRESHOLD, "orta": MEDIUM_RISK_THRESHOLD},
        "kohort_standardizasyonu": {
            "yks": YKS_COHORT_MEAN is not None and YKS_COHORT_STD is not None,
            "lise_diploma": LISE_COHORT_MEAN is not None and LISE_COHORT_STD is not None,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8000),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
