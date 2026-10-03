
import os

import requests
import streamlit as st

st.set_page_config(page_title="Öğrenci Terk Riski", page_icon="🎓", layout="wide")

DEFAULT_API = os.getenv("API_URL", "http://localhost:8000")
KYK = {"Yok": "yok", "Yalnızca kredi": "kredi", "Yalnızca burs": "burs", "Kredi ve burs": "kredi_ve_burs"}
BARINMA = {"Ailesinin yanında": "aile_yaninda", "KYK yurdu": "kyk_yurdu", "Özel yurt": "ozel_yurt", "Kirada": "kirada"}
RISK_STYLE = {"yuksek": ("🔴", st.error), "orta": ("🟡", st.warning), "dusuk": ("🟢", st.success)}
SEVERITY_ICON = {"yuksek": "🔴", "orta": "🟡", "bilgi": "🔵"}

with st.sidebar:
    st.header("⚙️ Bağlantı")
    api_url = st.text_input("API adresi", DEFAULT_API).rstrip("/")
    try:
        health = requests.get(f"{api_url}/health", timeout=5).json()
        if health.get("model_hazir"):
            st.success("API ve model hazır")
        else:
            st.error(f"Model yüklü değil: {(health.get('hata') or {}).get('mesaj', '?')}")
    except requests.RequestException:
        st.error("API'ye ulaşılamıyor. `uvicorn app:app --port 8000` çalışıyor mu?")
    st.caption(
        "Model, Portekiz verisiyle eğitilmiştir. KYK, harçlık, memnuniyet ve çalışma bilgileri "
        "yalnızca danışman uyarısı üretir; olasılığı değiştirmez."
    )

st.title("🎓 Öğrenci Terk Riski Erken Uyarı Sistemi")
tab_predict, tab_importance = st.tabs(["Öğrenci Analizi", "Özellik Önemi"])

with tab_predict:
    with st.form("form"):
        st.subheader("📚 Akademik Durum")
        c1, c2 = st.columns(2)
        sem = {}
        for col, key, title in ((c1, "donem_1", "1. Dönem (Güz)"), (c2, "donem_2", "2. Dönem (Bahar)")):
            with col:
                st.markdown(f"**{title}**")
                enrolled = st.number_input("Alınan ders", 0, 30, 6, key=f"{key}_a")
                passed = st.number_input("Geçilen ders", 0, 30, 5, key=f"{key}_g")
                mode = st.radio("Not girişi", ["AGNO (0-4)", "Vize/Final"], horizontal=True, key=f"{key}_m")
                if mode == "AGNO (0-4)":
                    agno = st.number_input("Dönem AGNO", 0.0, 4.0, 2.5, 0.01, key=f"{key}_agno")
                    grades = {"agno": agno}
                else:
                    vize = st.number_input("Vize ortalaması (0-100)", 0.0, 100.0, 55.0, key=f"{key}_v")
                    final = st.number_input("Final ortalaması (0-100)", 0.0, 100.0, 60.0, key=f"{key}_f")
                    weight = st.slider("Vize ağırlığı", 0.0, 1.0, 0.4, 0.05, key=f"{key}_w")
                    grades = {"vize_ortalamasi": vize, "final_ortalamasi": final, "vize_agirligi": weight}
                absent = st.number_input("Devamsızlıktan kalan ders", 0, 30, 0, key=f"{key}_d")
                sem[key] = {"alinan_ders": enrolled, "gecilen_ders": passed, "devamsiz_ders": absent, **grades}

        st.subheader("👤 Öğrenci ve Kayıt Bilgileri")
        a, b, c = st.columns(3)
        with a:
            age = st.number_input("Kayıt yaşı", 15, 80, 19)
            gender = st.radio("Cinsiyet", ["Kadın", "Erkek"], horizontal=True)
            foreign = st.checkbox("Yabancı uyruklu")
            other_city = st.checkbox("Ailesinden farklı şehirde okuyor")
        with b:
            uni = st.radio("Üniversite türü", ["Devlet", "Vakıf"], horizontal=True)
            tur = st.radio("Öğretim türü", ["Birinci öğretim", "İkinci öğretim"], horizontal=True)
            pref = st.number_input("YKS tercih sırası", 1, 30, 1)
        with c:
            yks = st.number_input("YKS yerleştirme puanı (0 = girilmedi)", 0.0, 560.0, 0.0, 0.5)
            lise = st.number_input("Lise diploma notu (0 = girilmedi)", 0.0, 100.0, 0.0, 0.1)

        st.subheader("💰 Mali ve Sosyal Durum")
        d, e, f = st.columns(3)
        with d:
            kyk = st.selectbox("KYK durumu", list(KYK))
            bar = st.selectbox("Barınma", list(BARINMA))
            other_burs = st.checkbox("KYK dışı burs var")
            vakif_burs = st.select_slider("Vakıf burs oranı (%)", [0, 25, 50, 75, 100], 0) if uni == "Vakıf" else 0
        with e:
            tuition = st.radio("Harç/öğrenim ücreti güncel mi?", ["Bilinmiyor / harç yok", "Evet", "Hayır"])
            debtor = st.checkbox("Vadesi geçmiş okul/yurt borcu var")
            income = st.number_input("Aylık gelir/harçlık (TL, 0 = girilmedi)", 0, 1_000_000, 0, 500)
            expense = st.number_input("Aylık zorunlu gider (TL, 0 = girilmedi)", 0, 1_000_000, 0, 500)
        with f:
            works = st.checkbox("Kısmi zamanlı çalışıyor")
            hours = st.number_input("Haftalık çalışma saati", 0, 80, 0) if works else 0
            sat = st.select_slider("Bölüm memnuniyeti (1 = çok düşük, 5 = çok yüksek)", [1, 2, 3, 4, 5], 3)

        submitted = st.form_submit_button("🔍 Analiz Et", type="primary")

    if submitted:
        mali = {
            "kyk_durumu": KYK[kyk], "barinma": BARINMA[bar], "diger_burs": other_burs,
            "vakif_burs_orani": vakif_burs, "borclu": debtor, "kismi_zamanli_calisiyor": works,
        }
        if tuition != "Bilinmiyor / harç yok":
            mali["harc_odemesi_guncel"] = tuition == "Evet"
        if income > 0 and expense > 0:
            mali.update(aylik_gelir_try=income, aylik_gider_try=expense)
        if works and hours > 0:
            mali["haftalik_calisma_saati"] = hours
        giris = {"universite_turu": uni.lower().replace("ı", "i"), "tercih_sirasi": pref,
                 "ogretim_turu": "birinci" if tur.startswith("Birinci") else "ikinci"}
        if yks > 0:
            giris["yks_yerlestirme_puani"] = max(yks, 100.0)
        if lise > 0:
            giris["lise_diploma_notu"] = max(lise, 50.0)
        payload = {
            "ogrenci": {"kayit_yasi": age, "cinsiyet": gender.lower().replace("ı", "i"),
                        "uyruk": "yabanci" if foreign else "tc", "ailesinden_farkli_sehirde": other_city},
            "giris": giris, "akademik": sem, "mali": mali, "anket": {"bolum_memnuniyeti": sat},
        }
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)
            body = resp.json()
        except (requests.RequestException, ValueError) as exc:
            st.error(f"API çağrısı başarısız: {exc}")
            st.stop()

        if resp.status_code != 200:
            err = body.get("hata", {})
            st.error(f"{err.get('mesaj', 'Bilinmeyen hata')} (HTTP {resp.status_code})")
            for item in err.get("detay") or []:
                if isinstance(item, dict):
                    st.write(f"• **{item.get('alan')}**: {item.get('mesaj')}")
            st.stop()

        pred = body["tahmin"]
        icon, box = RISK_STYLE[pred["risk_seviyesi"]]
        box(f"{icon} **{pred['risk_seviyesi_tr']}** - en olası durum: {pred['sinif_tr']}")
        cols = st.columns(3)
        for col, (name, label) in zip(cols, (("Dropout", "Terk"), ("Enrolled", "Kayıtlı"), ("Graduate", "Mezun"))):
            p = pred["olasiliklar"][name]
            col.metric(f"{label} olasılığı", f"{p * 100:.1f}%")
            col.progress(float(p))

        adv = body["danisman_uyarilari"]
        st.subheader("🧭 Danışman Uyarıları (model dışı)")
        if adv["manuel_inceleme_onerilir"]:
            st.info("Danışman değerlendirmesi önerilir.")
        for a_ in adv["uyarilar"] or []:
            st.write(f"{SEVERITY_ICON.get(a_['seviye'], '•')} {a_['mesaj']}")
        if not adv["uyarilar"]:
            st.write("Ek uyarı yok.")
        if body["uyarilar"]:
            with st.expander("Veri kalitesi uyarıları"):
                for w in body["uyarilar"]:
                    st.write(f"• {w}")
        with st.expander("Varsayımlar"):
            for w in body["varsayimlar"]:
                st.write(f"• {w}")
        st.caption(body["sorumluluk_notu"])

with tab_importance:
    top_k = st.slider("Gösterilecek özellik sayısı", 5, 30, 12)
    if st.button("Özellik önemini getir"):
        try:
            data = requests.get(f"{api_url}/feature-importance", params={"top_k": top_k}, timeout=30).json()
            rows = [{"Sıra": i["sira"], "Özellik": i["turkce_ad"], "Önem %": i["onem_yuzde"],
                     "Türkiye uyumu": i["turkiye_uyumu"], "Bağlam": i["turkiye_baglami"]}
                    for i in data["en_onemli"]]
            st.bar_chart({r["Özellik"]: r["Önem %"] for r in rows}, horizontal=True)
            st.dataframe(rows, hide_index=True)
            for line in data["yorum"]:
                st.write(f"• {line}")
        except (requests.RequestException, KeyError, ValueError) as exc:
            st.error(f"Rapor alınamadı: {exc}")
