#!/bin/bash

SCRIPT="SuLuSa_pipeline_monthly_fixed.py"
COUNTRY_CSV="HDI_GPI_with_GDP_LATLON.csv"

declare -A PANELS
PANELS[ukraine]="ukraine_Panel.csv"
PANELS[mexico]="mexico_panel.csv"
PANELS[syria]="syria_panel.csv"

declare -A ARTICLES
ARTICLES[ukraine]="ukraine.csv"
ARTICLES[mexico]="mexico.csv"
ARTICLES[syria]="syria.csv"

declare -A ALPHA_LABELS
ALPHA_LABELS[0.001]="00_1"
ALPHA_LABELS[0.0005]="000_5"
ALPHA_LABELS[0.0001]="000_1"

for country in ukraine mexico syria
do
    for alpha in 0.001 0.0005 0.0001
    do
        label=${ALPHA_LABELS[$alpha]}
        output_dir="${country}_alpha_${label}"

        echo "======================================"
        echo "Running $country with alpha=$alpha"
        echo "Output folder: $output_dir"
        echo "======================================"

        python "$SCRIPT" \
            --article_csv "${ARTICLES[$country]}" \
            --panel_csv "${PANELS[$country]}" \
            --country_csv "$COUNTRY_CSV" \
            --output_dir "$output_dir" \
            --alpha "$alpha"
    done
done