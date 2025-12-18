# app.py
from __future__ import annotations
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Set

import pandas as pd
import streamlit as st


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
    - menghapus baris dengan No. Faktur / Barang kosong
    - normalisasi nama produk (lowercase + strip)
    - menghapus duplikasi item dalam faktur yang sama
    - mengagregasi menjadi 1 baris per faktur berisi list item unik
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
                lift = conf / sup_y
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
    rules_longtail,
    freq_main,
    params: AprioriParams,
):
    """Bangun 5 tier rekomendasi berdasarkan aturan utama + long tail."""
    # Tier 1: produk inti dengan support terbesar
    item_support = compute_item_support(freq_main)
    core_items_sorted = sorted(
        item_support.items(), key=lambda x: x[1], reverse=True
    )
    tier1_items = [name for name, _ in core_items_sorted[:3]]
    tier1_set = set(tier1_items)

    # Skor item global (semua aturan)
    item_scores = compute_item_scores(rules_main)

    # Tier 2: produk yang sering berpasangan dengan Tier 1 (partner bundling utama)
    partner_scores = defaultdict(float)
    for A, B, sup, conf, lift in rules_main:
        rule_items = set(A) | set(B)
        if tier1_set & rule_items:
            rule_score = sup * conf * lift
            for i in rule_items - tier1_set:
                partner_scores[i] += rule_score

    tier2_items = [
        item for item, _ in sorted(partner_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # Tier 3: produk pendukung di sekitar Tier 1 & 2
    exclude_tier12 = tier1_set | set(tier2_items)
    tier3_candidates = {
        item: score
        for item, score in item_scores.items()
        if item not in exclude_tier12
    }
    tier3_items = [
        item for item, _ in sorted(tier3_candidates.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # Tier 4: produk long tail ber-lift tinggi (dari aturan long tail)
    lt_scores = compute_item_scores(rules_longtail)
    tier4_items = [
        item for item, _ in sorted(lt_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # Tier 5: produk pelengkap, sering muncul sebagai consequent
    consequent_scores = defaultdict(float)
    for A, B, sup, conf, lift in rules_main:
        rule_score = sup * conf * lift
        for item in B:
            consequent_scores[item] += rule_score

    exclude_prev = tier1_set | set(tier2_items) | set(tier3_items) | set(tier4_items)
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


def rules_from_freq(freq, abs_counts, params: AprioriParams, longtail: bool = False):
    """
    Helper untuk membentuk DataFrame aturan:
    - main rules: hanya 2-itemset (filter_exact_2_only=True)
    - long tail: boleh 2+ item (filter_exact_2_only=False) dengan support di rentang long tail
    """
    if longtail:
        min_sup = params.longtail_min_support
        max_sup = params.longtail_max_support
        filter_2 = False      # long tail boleh lebih dari 2 item
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


# ==========================
# 3. Streamlit app
# ==========================

def main():
    st.title("Rekomendasi Penataan Produk Berbasis Advanced Apriori")

    st.write(
        "Upload satu atau beberapa dataset transaksi (.csv) dengan kolom "
        "`No. Faktur` dan `Barang`. Aplikasi akan melakukan preprocessing, "
        "menjalankan algoritma Advanced Apriori, dan menghasilkan rekomendasi "
        "tier produk untuk penataan rak dan paket bundling."
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

        st.subheader("Ringkasan Data Mentah")
        st.write(f"Total baris data mentah: **{len(df_all):,}**")

        # Preprocessing & agregasi
        baskets = aggregate_transactions(df_all)
        st.write(f"Total transaksi hasil agregasi: **{len(baskets):,}**")

        # Konversi ke list of set
        transactions = [set(items) for items in baskets["Barang"]]

        # Apriori utama (support 1%)
        freq_main, abs_main = apriori(transactions, params.min_support)
        rules_main_df, rules_main_list = rules_from_freq(
            freq_main, abs_main, params, longtail=False
        )

        # Apriori long tail (support 0.2%–1%)
        freq_lt, abs_lt = apriori(transactions, params.longtail_min_support)
        rules_lt_df, rules_lt_list = rules_from_freq(
            freq_lt, abs_lt, params, longtail=True
        )

        st.subheader("Ringkasan Aturan Asosiasi")
        st.write(
            f"Jumlah aturan utama (support ≥ {params.min_support:.3f}): "
            f"**{len(rules_main_df):,}**"
        )
        st.write(
            "Jumlah aturan long tail "
            f"(support {params.longtail_min_support:.3f}–{params.longtail_max_support:.3f}): "
            f"**{len(rules_lt_df):,}**"
        )

        # Bangun tier rekomendasi
        tiers = build_tiers(
            rules_main_list,
            rules_lt_list,
            freq_main,
            params,
        )

        st.subheader("Rekomendasi Tier Penataan Produk")

        st.markdown(
            "**Tier 1 – Blok utama rak**  \n"
            "Produk dengan penjualan dan keterkaitan paling kuat, sebaiknya ditempatkan di "
            "area rak yang paling strategis dan selalu dijaga ketersediaan stoknya: "
            f"**{format_list_id(tiers['tier1'])}**."
        )

        st.markdown(
            "**Tier 2 – Paket bundling utama**  \n"
            "Produk yang paling sering berpasangan dengan Tier 1 dan layak dijadikan paket bundling "
            "di rak yang sama atau di area display yang sangat berdekatan: "
            f"**{format_list_id(tiers['tier2'])}**."
        )

        st.markdown(
            "**Tier 3 – Produk pendukung di sekitar blok utama**  \n"
            "Produk yang sering muncul bersama Tier 1 dan Tier 2 dengan kekuatan asosiasi sedikit lebih rendah; "
            "cocok ditempatkan di rak sekitar blok utama untuk memperbesar peluang cross-selling: "
            f"**{format_list_id(tiers['tier3'])}**."
        )

        st.markdown(
            "**Tier 4 – Produk long tail berpotensi tinggi**  \n"
            "Produk dengan frekuensi penjualan rendah tetapi memiliki nilai lift tinggi pada aturan long tail; "
            "relevan untuk program paket tematik atau promosi khusus yang menonjolkan produk UMKM: "
            f"**{format_list_id(tiers['tier4'])}**."
        )

        st.markdown(
            "**Tier 5 – Produk pelengkap dan peluang di dekat kasir**  \n"
            "Produk yang sering muncul sebagai konsekuen dalam aturan asosiasi dan dapat berperan sebagai "
            "pelengkap keranjang belanja, sehingga cocok ditempatkan dekat kasir atau area display tambahan: "
            f"**{format_list_id(tiers['tier5'])}**."
        )

        # Contoh aturan utama
        st.subheader("Contoh Aturan Utama (Top 10 berdasarkan lift)")
        if not rules_main_df.empty:
            top_rules_display = rules_main_df.sort_values("lift", ascending=False).head(10).copy()
            top_rules_display["antecedent"] = top_rules_display["antecedent"].apply(
                lambda t: ", ".join(t)
            )
            top_rules_display["consequent"] = top_rules_display["consequent"].apply(
                lambda t: ", ".join(t)
            )
            st.dataframe(top_rules_display)
        else:
            st.write("Aturan utama belum terbentuk pada parameter yang digunakan.")

        # Contoh aturan long tail (semua, diurutkan berdasarkan lift)
        st.subheader("Contoh Aturan Long Tail (diurutkan berdasarkan lift)")
        if not rules_lt_df.empty:
            top_lt_display = rules_lt_df.sort_values("lift", ascending=False).copy()
            top_lt_display["antecedent"] = top_lt_display["antecedent"].apply(
                lambda t: ", ".join(t)
            )
            top_lt_display["consequent"] = top_lt_display["consequent"].apply(
                lambda t: ", ".join(t)
            )
            st.dataframe(top_lt_display)
        else:
            st.write("Aturan long tail belum terbentuk pada parameter yang digunakan.")


if __name__ == "__main__":
    main()
