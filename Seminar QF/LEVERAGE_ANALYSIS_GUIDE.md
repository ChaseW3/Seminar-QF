# Leverage Group Analysis - User Guide

## Overview

The `cds_correlation_analysis.ipynb` notebook has been enhanced with comprehensive leverage analysis capabilities. The notebook now:

1. **Calculates leverage ratios** for all firms
2. **Creates three leverage groups** (Low, Mid, High) using tercile splits
3. **Analyzes CDS correlation performance** separately for each leverage group
4. **Tests hypotheses** about whether models perform differently across leverage levels

## What Was Added

### Section 0: Leverage Analysis and Firm Grouping

**New Cells:**
1. **Load and Calculate Leverage** (Cell 4)
   - Loads merged data with equity and liability information
   - Calculates leverage ratio: `Liabilities / (Liabilities + Market Cap)`
   - Computes average leverage for each firm over full time period
   - Displays all 34 firms sorted by leverage

2. **Create Leverage Groups** (Cell 5)
   - Uses tercile split to create three equal-sized groups
   - Assigns each firm to: Low Leverage, Mid Leverage, or High Leverage
   - Shows group boundaries and lists firms in each group

3. **Visualize Leverage Distribution** (Cell 6)
   - 4-panel visualization:
     - Histogram with tercile boundaries
     - Box plot by group
     - Bar chart of all firms (color-coded by group)
     - Summary statistics table
   - Saves: `leverage_analysis.png`

### Section 3B: Correlation Analysis by Leverage Groups

**New Cells:**
4. **Merge Leverage Groups with Correlations** (Cell 16)
   - Joins leverage group assignments to correlation results
   - Calculates weighted average statistics for each group
   - Shows aggregate performance by leverage group

5. **Detailed Leverage Group Breakdown** (Cell 17)
   - For each leverage group:
     - Lists all firms in the group
     - Shows leverage range and average
     - Displays model performance (RMSE, r_levels, r_changes)
     - Tests statistical significance
     - Identifies best model for each metric

6. **Visualize Performance by Leverage** (Cell 18)
   - 4-panel comparison:
     - RMSE by leverage group (bar chart)
     - Correlation of levels by group (bar chart)
     - Correlation of changes by group (bar chart)
     - Heatmap of correlations
   - Saves: `cds_correlation_by_leverage.png`

7. **Firm-Level Details** (Cell 19)
   - Shows individual firms within each leverage group
   - Sorted by GARCH correlation performance
   - Displays key metrics for GARCH and MS-GARCH models

8. **Export Results** (Cell 20)
   - Creates: `cds_correlation_by_leverage_group.csv` (aggregate stats)
   - Creates: `cds_correlations_with_leverage.csv` (firm-level with groups)

9. **Key Questions and Hypotheses** (Cell 21)
   - Markdown explaining what to look for in results
   - Theory-driven expectations
   - Practical implications

## How to Use

### Step 1: Run the Notebook
```bash
# Open Jupyter or VS Code
cd "notebooks/"
# Open: cds_correlation_analysis.ipynb
# Run all cells sequentially
```

### Step 2: Review Leverage Distribution
After running cells 4-6, you'll see:
- Which firms are high/mid/low leverage
- Visual distribution of leverage ratios
- Summary statistics by group

**Key questions:**
- Are banks in the high leverage group? (Expected: Yes)
- Is leverage evenly distributed? (Expected: No, banks much higher)

### Step 3: Analyze Performance by Leverage
After running cells 16-18, review:
- Do models perform better for specific leverage groups?
- Is RMSE higher for high-leverage firms?
- Do correlations vary systematically with leverage?

### Step 4: Interpret Results

**Expected Patterns:**

1. **Low Leverage Firms** (bottom tercile)
   - **Better overall fit** - less default risk, cleaner signals
   - **Lower RMSE** - easier to model
   - **Higher correlations** - predictable CDS spreads
   - **Best models**: Classical Merton may work well

2. **Mid Leverage Firms** (middle tercile)
   - **Sweet spot for Merton model** - moderate default probability
   - **Balanced performance** across models
   - **Good correlations** for both levels and changes

3. **High Leverage Firms** (top tercile - mostly banks)
   - **Harder to model** - distress, illiquidity, regime changes
   - **Higher RMSE** - more noise in CDS spreads
   - **Lower correlations** - structural breaks, non-normality
   - **Best models**: MS-GARCH may outperform (captures regime switching)

**Hypotheses to Test:**

✅ **H1**: RMSE increases with leverage (high-leverage firms harder to model)
✅ **H2**: Correlation of levels highest for low-leverage firms
✅ **H3**: MS-GARCH performs relatively better for high-leverage firms
✅ **H4**: Banks (high leverage) show different patterns than corporates

## Output Files

### CSV Files
1. **`cds_correlation_by_leverage_group.csv`**
   - Aggregate statistics for each leverage group
   - Columns: Leverage Group, N Firms, Total Obs, Avg Leverage, Model RMSEs, Model Correlations
   - Use for: Summary tables, group comparisons

2. **`cds_correlations_with_leverage.csv`**
   - Firm-level correlations with leverage group assignments
   - Columns: All original correlation columns + avg_leverage + leverage_tercile
   - Use for: Detailed analysis, filtering by group, regressions

### Visualizations
1. **`leverage_analysis.png`**
   - 4-panel overview of leverage distribution
   - Use for: Understanding sample composition

2. **`cds_correlation_by_leverage.png`**
   - 4-panel comparison of model performance across groups
   - Use for: Main results figure, presentations

## Statistical Interpretation

### Significance Testing
All correlations are tested using:
- **Pearson correlation coefficient** with t-test
- **Fisher's z-transformation** for confidence intervals
- **Significance levels**: *** p<0.001, ** p<0.01, * p<0.05

### Sample Size Considerations
- Each leverage group has ~11-12 firms (tercile split)
- Weighted averages use total observations (weighted by n_obs)
- Approximate significance tests use pooled sample size
- **Interpretation**: Even small correlation differences can be significant due to large n

### Model Comparison Within Groups
The notebook compares:
1. **RMSE**: Which model has lowest prediction error?
2. **Correlation of Levels**: Which captures CDS spread levels best?
3. **Correlation of Changes**: Which captures daily dynamics best? (Byström 2006)

## Research Questions Addressed

### 1. Does leverage affect model performance?
- Compare RMSE across groups
- Test if high-leverage → higher RMSE
- Statistical test: ANOVA or Kruskal-Wallis on RMSE by group

### 2. Do different models work for different leverage profiles?
- Compare "best model" frequency across groups
- Is GARCH dominant everywhere or only for specific groups?
- MS-GARCH advantage for high-leverage firms?

### 3. Are financial firms (banks) fundamentally different?
- Banks typically in high-leverage group
- Do they show lower correlations?
- Structural model limitations for financials?

### 4. Practical implications
- **Portfolio managers**: Use model selection based on firm leverage
- **Risk managers**: Adjust confidence in CDS model for high-leverage names
- **Researchers**: Document leverage as key determinant of model fit

## Next Steps

### Further Analysis Ideas

1. **Regression Analysis**
   ```python
   # Regress correlation on leverage
   import statsmodels.api as sm
   
   X = sm.add_constant(results_df['avg_leverage'])
   y = results_df['GARCH_corr_lvl']
   model = sm.OLS(y, X).fit()
   print(model.summary())
   ```

2. **Time-Varying Leverage**
   - Currently uses average leverage over full period
   - Could analyze how leverage changes affect correlations dynamically

3. **Interaction Effects**
   - Leverage × Sector
   - Leverage × Market Conditions (crisis vs normal)

4. **Optimal Leverage Threshold**
   - Instead of terciles, find leverage level where model performance breaks down
   - ROC-style analysis

5. **Multi-Group Analysis**
   - 4-5 groups instead of 3
   - Allows more granular patterns

## Troubleshooting

### If groups are unbalanced:
- Check if tercile split worked (should be ~equal sizes)
- Some firms may have missing leverage data
- Verify merge on gvkey succeeded

### If correlations don't vary by leverage:
- May indicate leverage is not a key driver
- Other factors (sector, size, liquidity) may dominate
- Check if leverage range is sufficient (too narrow → no variation)

### If visualizations don't render:
- Ensure matplotlib and seaborn installed
- Check if output directory exists
- Try running cells individually

## References

- **Byström (2006)**: Correlation of changes to test dynamic fit
- **Merton (1974)**: Structural model theory
- **Hamilton (1989)**: Regime-switching models
- **Literature**: Structural models perform worse for highly leveraged firms

## Contact

For questions about the leverage analysis:
1. Review this guide
2. Check cell outputs for detailed explanations
3. Inspect generated CSV files
4. Review visualizations

**Key Insight**: Leverage is a fundamental determinant of default risk and should significantly affect how well structural credit models perform. This analysis quantifies that relationship empirically.
