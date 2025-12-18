# app.py
from __future__ import annotations
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Set

import pandas as pd
import streamlit as st


# ==========================
# 0. Page config & simple CSS
# ==========================

st.set_page_config(
    page_title="Product Recommender",
    page_icon="🛒",
    layout="wide",
)

# CSS untuk "bubble" produk dan kartu tier
st.markdown(
    """
    <style>
    .tier-section {
        padding: 1rem 1.2rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .tier-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.25rem;
    }
    .tier-desc {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .badge {
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(4px);
    }
    .badge.t1 {background-color: rgba(255, 193, 7, 0.25);}
    .badge.t2 {background-color: rgba(40, 167, 69, 0.25);}
    .badge.t3 {background-color: rgba(0, 123, 255, 0.25);}
    .badge.t4 {background-color: rgba(220, 53, 69, 0.25);}
    .badge.t5 {background-color: rgba(111, 66, 193, 0.25);}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================
# 1. Parameter dan fungsi Apriori
# ==========================

@dataclass
class AprioriParams:
    min_support: float = 0.01
    min_confidence: float = 0.30
    min_lift: float = 1.5
    longtail_min_support: float = 0.002
    longtail_max_support: float = 0.01


def aggregate_transactions(
    df: pd.DataFrame,
    invoice_col: str = "No. Faktur",
    item_col: str = "Barang",
) -> pd.DataFrame:
    """
    Preprocessing utama:
    - hapus baris dengan No. Faktur / Barang kosong
    - normalisasi nama produk (lowercase + strip)
    - hapus duplikasi item dalam faktur yang sama
    - agregasi menjadi 1 baris per faktur berisi list item unik
    """
    df = df.dropna(subset=[invoice_col, item_col]).copy()
    df[item_col] = df[item_col].astype(str).str.lower().str.strip()
    df = df.drop_duplicates(subset=[invoice_col, item_col])

    baskets = (
        df.groupby(invoice_col)[item_col]
        .apply(lambda items: sorted(set(items)))
        .reset_index(name="Barang")
    )
    return baskets


def apriori(transactions: List[Set[str]], min_support: float):
    """Algoritma Apriori sederhana untuk frequent itemset."""
    n = len(transactions)
    min_count = max(1, int(min_support * n))
    item_counts = defaultdict(int)

    # 1-itemset
    for t in transactions:
        for i in t:
            item_counts[frozenset([i])] += 1

    freq = {i: c / n for i, c in item_counts.items() if c >= min_count}
    abs_counts = {i: c for i, c in item_counts.items() if c >= min_count}
    L = abs_counts.copy()
    k = 2

    # k-itemset (k ≥ 2)
    while L:
        candidates = {
            i1 | i2
            for i1 in L for i2 in L
            if len(i1 | i2) == k
        }

        next_counts = defaultdict(int)
        for t in transactions:
            for c in candidates:
                if c.issubset(t):
                    next_counts[c] += 1

        L = {
            i: count
            for i, count in next_counts.items()
            if count >= min_count
        }

        if not L:
            break

        freq.update({i: c / n for i, c in L.items()})
        abs_counts.update(L)
        k += 1

    return freq, abs_counts


def generate_rules(
    freq,
    abs_counts,
    min_conf: float,
    min_lift: float,
    filter_exact_2_only: bool = False,
):
    """
    Membentuk aturan asosiasi dari frequent itemset.
    Jika filter_exact_2_only=True maka hanya aturan dengan 2 item (A -> B) yang diambil.
    """
    rules = []
    for itemset, sup_xy in freq.items():
        if len(itemset) < 2:
            continue
        if filter_exact_2_only and len(itemset) != 2:
            continue

        for r in range(1, len(itemset)):
            for A in itertools.combinations(itemset, r):
                A = frozenset(A)
                B = itemset - A
                sup_x = freq.get(A, 0)
                sup_y = freq.get(B, 0)
                if sup_x == 0 or sup_y == 0:
                    continue
                conf = sup_xy / sup_x
                lift = sup_xy / (sup_x * sup_y)  # sama dengan conf/sup_y
                if conf >= min_conf and lift >= min_lift:
                    rules.append((tuple(sorted(A)), tuple(sorted(B)), sup_xy, conf, lift))
    return rules


# ==========================
# 2. Fungsi analisis bisnis → Tier
# ==========================

def compute_item_support(freq):
    """Ambil support untuk item tunggal dari frequent itemset."""
    support_1 = {}
    for itemset, sup in freq.items():
        if len(itemset) == 1:
            item = next(iter(itemset))
            support_1[item] = sup
    return support_1


def compute_item_scores(rules):
    """Skor pentingnya item berdasarkan kontribusinya di semua aturan."""
    scores = defaultdict(float)
    for A, B, sup, conf, lift in rules:
        rule_score = sup * conf * lift
        for item in set(A) | set(B):
            scores[item] += rule_score
    return scores


def format_list_id(items, max_items=5):
    """Format list nama produk jadi kalimat pendek."""
    items = list(items)[:max_items]
    if not items:
        return "-"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} dan {items[1]}"
    return f"{', '.join(items[:-1])}, dan {items[-1]}"


def build_tiers(
    rules_main,
    rules_potential,
    freq_main,
    freq_potential,
    params: AprioriParams,
):
    """
    Bangun 5 tier rekomendasi berdasarkan gabungan aturan utama + aturan produk potensial.
    - Tier 1: produk terkuat secara global
    - Tier 2: partner bundling terkuat untuk Tier 1
    - Tier 3: produk pendukung lain di sekitar Tier 1 & 2
    - Tier 4: produk potensial (support relatif rendah, lift tinggi)
    - Tier 5: produk pelengkap yang sering muncul sebagai consequent
    """
    # Gabungkan frequent itemset untuk support item tunggal
    freq_combined = {}
    freq_combined.update(freq_potential)
    freq_combined.update(freq_main)
    item_support = compute_item_support(freq_combined)

    # Gabungkan semua aturan
    rules_all = list(rules_main) + list(rules_potential)

    # Skor item global
    item_scores_all = compute_item_scores(rules_all)

    # -------- Tier 1: produk inti (skor tertinggi) --------
    core_sorted = sorted(item_scores_all.items(), key=lambda x: x[1], reverse=True)
    tier1_items = [name for name, _ in core_sorted[:3]]
    tier1_set = set(tier1_items)

    # -------- Tier 2: partner bundling utama untuk Tier 1 --------
    partner_scores = defaultdict(float)
    for A, B, sup, conf, lift in rules_all:
        rule_items = set(A) | set(B)
        if tier1_set & rule_items:
            rule_score = sup * conf * lift
            for i in rule_items - tier1_set:
                partner_scores[i] += rule_score

    tier2_items = [
        item for item, _ in sorted(partner_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # -------- Tier 3: produk pendukung di sekitar blok utama --------
    exclude_tier12 = tier1_set | set(tier2_items)
    tier3_candidates = {
        item: score
        for item, score in item_scores_all.items()
        if item not in exclude_tier12
    }
    tier3_items = [
        item for item, _ in sorted(tier3_candidates.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # -------- Tier 4: produk potensial (support rendah, lift tinggi) --------
    pot_scores = compute_item_scores(rules_potential)
    exclude_tier123 = exclude_tier12 | set(tier3_items)
    tier4_candidates = {
        item: score
        for item, score in pot_scores.items()
        if item not in exclude_tier123
    }
    tier4_items = [
        item for item, _ in sorted(tier4_candidates.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # -------- Tier 5: produk pelengkap (sering jadi consequent) --------
    consequent_scores = defaultdict(float)
    for A, B, sup, conf, lift in rules_all:
        rule_score = sup * conf * lift
        for item in B:
            consequent_scores[item] += rule_score

    exclude_prev = exclude_tier123 | set(tier4_items)
    tier5_candidates = {
        item: score
        for item, score in consequent_scores.items()
        if item not in exclude_prev
    }
    tier5_items = [
        item for item, _ in sorted(tier5_candidates.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    tiers = {
        "tier1": tier1_items,
        "tier2": tier2_items,
        "tier3": tier3_items,
        "tier4": tier4_items,
        "tier5": tier5_items,
    }
    return tiers


def rules_from_freq(freq, abs_counts, params: AprioriParams, potential: bool = False):
    """
    Helper untuk membentuk DataFrame aturan:
    - main rules: hanya 2-itemset (filter_exact_2_only=True)
    - produk potensial: boleh 2+ item (filter_exact_2_only=False) dengan support di rentang rendah
    """
    if potential:
        min_sup = params.longtail_min_support
        max_sup = params.longtail_max_support
        filter_2 = False      # produk potensial boleh lebih dari 2 item
    else:
        min_sup = params.min_support
        max_sup = 1.0
        filter_2 = True       # aturan utama fokus 2-item

    rules = generate_rules(
        freq,
        abs_counts,
        min_conf=params.min_confidence,
        min_lift=params.min_lift,
        filter_exact_2_only=filter_2,
    )

    filtered = [
        r for r in rules
        if min_sup <= r[2] <= max_sup
    ]

    df = pd.DataFrame(
        filtered,
        columns=["antecedent", "consequent", "support", "confidence", "lift"]
    )
    return df, filtered


def render_tier(name: str, desc: str, items: List[str], tier_class: str):
    """Render satu blok tier dengan bubble produk."""
    if items:
        bubbles = "".join(
            f'<span class="badge {tier_class}">{p}</span>' for p in items
        )
    else:
        bubbles = '<em>Tidak ada produk teridentifikasi pada tier ini.</em>'

    st.markdown(
        f"""
        <div class="tier-section">
          <div class="tier-title">{name}</div>
          <div class="tier-desc">{desc}</div>
          <div class="badge-row">
            {bubbles}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================
# 3. Streamlit app
# ==========================

def main():
    st.header("🛒 Rekomendasi Penataan Produk")
    st.caption(
        "Analisis market basket otomatis dengan algoritma Advanced Apriori "
        "untuk membantu penataan rak, bundling, dan promosi produk."
    )

    with st.expander("Petunjuk singkat", expanded=False):
        st.markdown(
            "- Siapkan file transaksi dalam format CSV dengan kolom **`No. Faktur`** dan **`Barang`**.\n"
            "- `No. Faktur` berisi nomor struk, `Barang` berisi nama produk.\n"
            "- Upload satu atau beberapa file sekaligus, lalu klik tombol **Proses Analisis**.\n"
            "- Di bawah akan muncul rekomendasi Tier 1–5 dan contoh aturan asosiasi."
        )

    uploaded_files = st.file_uploader(
        "Upload file transaksi (boleh lebih dari satu)",
        type="csv",
        accept_multiple_files=True,
    )

    params = AprioriParams()

    if not uploaded_files:
        st.info("Silakan upload minimal satu file CSV untuk mulai analisis.")
        return

    if st.button("Proses Analisis"):
        # Gabung semua file yang di-upload
        dfs = []
        for f in uploaded_files:
            df = pd.read_csv(f)
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)

        st.subheader("✨ Ringkasan Data")
        st.write(f"Total baris data mentah: **{len(df_all):,}**")

        # Preprocessing & agregasi
        baskets = aggregate_transactions(df_all)
        st.write(f"Total transaksi hasil agregasi: **{len(baskets):,}**")

        # Konversi ke list of set
        transactions = [set(items) for items in baskets["Barang"]]

        # Apriori utama (support 1%)
        freq_main, abs_main = apriori(transactions, params.min_support)
        rules_main_df, rules_main_list = rules_from_freq(
            freq_main, abs_main, params, potential=False
        )

        # Apriori produk potensial (support 0.2%–1%)
        freq_pot, abs_pot = apriori(transactions, params.longtail_min_support)
        rules_pot_df, rules_pot_list = rules_from_freq(
            freq_pot, abs_pot, params, potential=True
        )

        st.subheader("📊 Ringkasan Aturan Asosiasi")
        st.write(
            f"Aturan utama (support ≥ {params.min_support:.3f}): "
            f"**{len(rules_main_df):,} aturan**"
        )
        st.write(
            f"Aturan produk potensial (support {params.longtail_min_support:.3f}"
            f"–{params.longtail_max_support:.3f}): "
            f"**{len(rules_pot_df):,} aturan**"
        )

        # Bangun tier rekomendasi dari gabungan aturan
        tiers = build_tiers(
            rules_main_list,
            rules_pot_list,
            freq_main,
            freq_pot,
            params,
        )

        st.subheader("🎯 Rekomendasi Tier Penataan Produk")

        render_tier(
            "Tier 1 – Blok utama rak",
            (
                "Produk dengan kekuatan asosiasi dan kontribusi penjualan tertinggi. "
                "Disarankan ditempatkan di area rak paling strategis dan selalu dijaga stoknya."
            ),
            tiers["tier1"],
            "t1",
        )

        render_tier(
            "Tier 2 – Paket bundling utama",
            (
                "Produk yang paling sering berpasangan dengan Tier 1 dan layak dijadikan "
                "paket bundling di rak yang sama atau di area display yang berdekatan."
            ),
            tiers["tier2"],
            "t2",
        )

        render_tier(
            "Tier 3 – Produk pendukung di sekitar blok utama",
            (
                "Produk yang sering muncul bersama Tier 1 dan Tier 2 dengan kekuatan asosiasi "
                "sedikit lebih rendah, cocok ditempatkan di sekitar blok utama untuk "
                "memperbesar peluang cross-selling."
            ),
            tiers["tier3"],
            "t3",
        )

        render_tier(
            "Tier 4 – Produk potensial untuk promosi",
            (
                "Produk dengan frekuensi penjualan relatif rendah namun memiliki nilai asosiasi "
                "kuat dalam keranjang tertentu. Cocok untuk program paket tematik atau "
                "promosi khusus yang menonjolkan produk."
            ),
            tiers["tier4"],
            "t4",
        )

        render_tier(
            "Tier 5 – Produk pelengkap dan peluang di dekat kasir",
            (
                "Produk yang sering muncul sebagai konsekuen dalam aturan asosiasi dan dapat "
                "berperan sebagai pelengkap keranjang belanja, cocok ditempatkan dekat kasir "
                "atau area display tambahan."
            ),
            tiers["tier5"],
            "t5",
        )

        # ========= Penjelasan istilah pada tabel =========
        with st.expander("Penjelasan istilah pada tabel", expanded=False):
            st.markdown(
                """
                - **Antecedent**: produk yang terlebih dahulu ada di keranjang belanja  
                  (kondisi “jika pelanggan membeli ...”).
                - **Consequent**: produk lain yang cenderung ikut terbeli setelah antecedent muncul  
                  (bagian “maka juga membeli ...”).
                - **Support**: proporsi transaksi yang berisi kombinasi antecedent dan consequent.  
                  Makin besar support, makin sering pasangan produk itu muncul di data.
                - **Confidence**: peluang pelanggan membeli consequent **ketika** sudah membeli antecedent.  
                  Nilai tinggi berarti aturan tersebut cukup dapat dipercaya untuk rekomendasi.
                - **Lift**: seberapa kuat hubungan dua produk dibandingkan jika dibeli secara acak.  
                  Nilai lift di atas 1 menunjukkan kedua produk saling menguatkan satu sama lain.
                """
            )

        # Aturan utama
        st.subheader("🔎 Pasangan Produk / Aturan Utama")
        if not rules_main_df.empty:
            top_rules_display = rules_main_df.sort_values("lift", ascending=False).head(10).copy()
            top_rules_display["antecedent"] = top_rules_display["antecedent"].apply(
                lambda t: ", ".join(t)
            )
            top_rules_display["consequent"] = top_rules_display["consequent"].apply(
                lambda t: ", ".join(t)
            )
            st.dataframe(top_rules_display, use_container_width=True)
        else:
            st.write("Aturan utama belum terbentuk pada parameter yang digunakan.")

        # Aturan produk potensial
        st.subheader("🌱 Pasangan Produk / Aturan Potensial")
        if not rules_pot_df.empty:
            top_pot_display = rules_pot_df.sort_values("lift", ascending=False).copy()
            top_pot_display["antecedent"] = top_pot_display["antecedent"].apply(
                lambda t: ", ".join(t)
            )
            top_pot_display["consequent"] = top_pot_display["consequent"].apply(
                lambda t: ", ".join(t)
            )
            st.dataframe(top_pot_display, use_container_width=True)
        else:
            st.write("Aturan produk potensial belum terbentuk pada parameter yang digunakan.")


if __name__ == "__main__":
    main()
