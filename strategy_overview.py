
import streamlit as st

# ==============================================
# STRATEGY OVERVIEW PAGE
# ==============================================

def display_strategy_overview():
    st.title("Strategy Overview")
    st.markdown("## Table of Contents")
    st.markdown("- [Backtesting Methodology](#Backtesting-Methodology)")
    st.markdown("- [Strategy 1 Detailed Explanation](#strategy-1-detailed-explanation)")
    st.markdown("- [Strategy 2 Detailed Explanation](#strategy-2-detailed-explanation)")
    st.markdown("- [Strategy 3 Detailed Explanation](#strategy-3-detailed-explanation)")
    st.markdown("---")

    # Backtesting Methodology Section
    st.markdown("### Backtesting Methodology")
    st.markdown(
        """
### 🚀 Handling Different Trading Strategies

Each trading strategy affects the **portfolio value** differently. We now handle **three scenarios** based on the chosen strategy for each day.

#### 🟢 1️⃣ Case 1: Long YMAG (No Hedge)
##### **Condition:**
- If **VIX < 20** and **VVIX < 100**, we go **fully long on YMAG (without hedging QQQ).**

##### **Portfolio Update Formula:**
"""
    )
    st.latex(r"""\text{Portfolio Value}_t = \text{Shares Held} \times (\text{YMAG Price}_t + \text{YMAG Dividend}_t)""")
    st.markdown(
        """
Where:
- **Shares Held** = Number of shares purchased using the previous day's portfolio value:
"""
    )
    st.latex(r"""\text{Shares Held} = \frac{\text{Portfolio Value}_{t-1}}{\text{YMAG Price}_{t-1}}""")
    st.markdown(
        """
- **YMAG Price** = The price of YMAG at time $t$.
- **YMAG Dividend** = Dividend per share distributed at time $t$.

#### ✅ Example Calculation:
- **Yesterday’s Portfolio Value:** $10,000
- **YMAG Price Yesterday:** $20.00
- **Shares Purchased:** $10,000 / $20.00 = **500 shares**
- **Today’s YMAG Price:** $20.50
- **Dividend Today:** $0.50 per share
"""
    )
    st.latex(r"""\text{Portfolio Value}_t = 500 \times (20.50 + 0.50) = 500 \times 21.00 = 10,500""")
    st.markdown(
        """
---
#### 🔵 2️⃣ Case 2: Long YMAG + Short QQQ (Hedged)
##### **Condition:**
- If **VIX ≥ 20** or **VVIX ≥ 100**, we go **long YMAG and hedge by shorting QQQ**.
- Shorting QQQ means that **when QQQ goes up, we lose money**, and **when QQQ goes down, we gain money**.

##### **Portfolio Update Formula:**
"""
    )
    st.latex(r"""\text{Portfolio Value}_t = (\text{Shares Held} \times (\text{YMAG Price}_t + \text{YMAG Dividend}_t)) - \text{QQQ Hedge PnL}_t""")
    st.markdown(
        """
Where:
- **Shares Held** = Same as in Case 1:
"""
    )
    st.latex(r"""\text{Shares Held} = \frac{\text{Portfolio Value}_{t-1}}{\text{YMAG Price}_{t-1}}""")
    st.markdown(
        """
- **QQQ Hedge Profit/Loss (PnL):**
"""
    )
    st.latex(r"""\text{QQQ Hedge PnL}_t = \text{QQQ Shares Shorted} \times (\text{QQQ Price}_{t-1} - \text{QQQ Price}_t)""")
    st.markdown(
        """
- **QQQ Shares Shorted:** (Calculated only when the hedge is first applied)
"""
    )
    st.latex(r"""\text{QQQ Shares Shorted} = \frac{\text{Portfolio Value}_{t-1}}{\text{QQQ Price}_{t-1}}""")
    st.markdown(
        """
##### ✅ Example Calculation:
- **Yesterday’s Portfolio Value:** $10,000
- **YMAG Price Yesterday:** $20.00
- **Shares Purchased:** $10,000 / $20.00 = **500 shares**
- **Today’s YMAG Price:** $20.60
- **Dividend Today:** $0.50 per share
- **QQQ Price Yesterday:** $400 → **Today:** $405

##### **Step 1: Calculate Hedge PnL**
- **QQQ Shares Shorted:**
"""
    )
    st.latex(r"""\frac{10,000}{400} = 25 \text{ shares}""")
    st.markdown(
        """
- **QQQ Hedge Loss:**
"""
    )
    st.latex(r"""25 \times (400 - 405) = 25 \times (-5) = -125""")
    st.markdown(
        """
##### **Step 2: Update Portfolio Value**
"""
    )
    st.latex(r"""\text{Portfolio Value}_t = (500 \times (20.60 + 0.50)) - (-125)""")
    st.latex(r"""= (500 \times 21.10) - (-125) = 10,550 + 125 = 10,675""")
    st.markdown(
        """
---
#### 🔴 3️⃣ Case 3: No Investment (Stay in Cash)
##### **Condition:**
- If **VIX ≥ 20 or VVIX ≥ 100** and **correlation of YMAG with VIX or VVIX < -0.3**, we **do not invest**.
- The **portfolio remains unchanged**.

##### **Portfolio Update Formula:**
"""
    )
    st.latex(r"""\text{Portfolio Value}_t = \text{Portfolio Value}_{t-1}""")
    st.markdown(
        """
##### ✅ Example Calculation:
- **Yesterday’s Portfolio:** $10,000
- **Today’s Strategy:** `"No Investment"`
- **Portfolio Value Stays the Same:**
"""
    )
    st.latex(r"""\text{Portfolio Value}_t = 10,000""")
    st.markdown(
        """
---
#### 📌 **Summary Table**

| **Strategy**                     | **Formula Used** |
|-----------------------------------|------------------|
| **Long YMAG (No Hedge)**          | Portfolio Value_t = Shares Held × (YMAG Price_t + YMAG Dividend_t) $$ |
| **Long YMAG + Short QQQ**         | Portfolio Value_t = (Shares Held × (YMAG Price_t + YMAG Dividend_t)) - QQQ Hedge PnL_t  |
| **No Investment (Stay in Cash)**  | Portfolio Value_t = Portfolio Value_{t-1} |

This breakdown ensures the correct handling of **portfolio value updates under each trading strategy**, including the **correct hedge profit/loss for QQQ shorting**. 🚀

---
"""
    )
    st.markdown("### Strategy 1 Detailed Explanation")
    st.markdown(
        """
**Investment Rules for Strategy 1:**
1. **Long (No Hedge)** if VIX < 20 and VVIX < 100.  
2. **Long + Short QQQ** if VIX ≥ 20 or VVIX ≥ 100, provided the correlation of YMAX/YMAG with VIX/VVIX is not too negative (≥ -0.3).  
3. **No Investment** if VIX ≥ 20 or VVIX ≥ 100 **and** correlation < -0.3.  

**Default Values:**
- **VIX threshold:** 20  
- **VVIX threshold:** 100  
- **Correlation threshold:** -0.3  
- **Correlation window (days):** 14  

**Entry/Exit Details:**
- You enter a **long position** (fully invested) when volatility is low (VIX < 20, VVIX < 100).  
- If volatility picks up (VIX ≥ 20 or VVIX ≥ 100) but correlation is not too negative, you **hedge** by shorting QQQ.  
- If that same high-volatility scenario has a negative correlation < -0.3, you **exit** (stay in cash).  

**Sliders:**
- A **Correlation Window** slider (1–30 days) lets you adjust how many days are used to compute rolling correlations.  
- (Internally, you can also adjust VIX/VVIX thresholds if you incorporate additional sliders for them, but for now we 
do not do that as we have that already in strategy 2.)
    
---
    """
    )
    st.markdown("### Strategy 2 Detailed Explanation")
    st.markdown(
        """
**Investment Rules for Strategy 2:**
1. **Remain in market if**: 15 ≤ VIX ≤ 20,  90 ≤ VVIX < 100. That is VIX is within [15, 20] and VVIX is within [90, 100).
2. **Exit if**: VIX < 15 or VIX > 20 or VVIX < 90 or VVIX ≥ 100
3. **Re-Enter if**: VIX ∈ [15,20] and VVIX ∈ [90,95]

**Default Values (Sliders):**
- **VIX Lower:** 15  
- **VIX Upper:** 20  
- **VVIX Lower:** 90  
- **VVIX Upper:** 100  
- **VVIX Re-Entry Upper:** 95  

**Summary of Logic under Default Values**:
- In-Market Condition: 15 ≤ VIX ≤ 20, 90 ≤ VVIX < 100
- Exit Condition: VIX < 15 or VIX > 20, or VVIX < 90 or VVIX ≥ 100
- Re-Entry Condition: VIX ∈ [15,20], VVIX ∈ [90,95]

**Entry/Exit Details:**
- The strategy stays **in the market** only if volatility (VIX) is in a “safer” band (15–20) and VVIX is below 100 but above 90.  
- If volatility moves **outside** those bounds (e.g., VIX < 15, VIX > 20, VVIX < 90, or VVIX ≥ 100), it **exits** (goes to cash).  
- Once it exits, it won’t **re-enter** until both VIX and VVIX come back within a narrower range (VIX ∈ [15, 20], VVIX ∈ [90, 95]).  

**Sliders:**
- **VIX Lower/Upper**: Adjust the allowed volatility band (e.g., 15–20 by default).  
- **VVIX Lower/Upper**: Set the normal range for VVIX (e.g., 90–100 by default).  
- **VVIX Re-Entry Upper**: The threshold for re-entering the market (default 95).

---
"""
    )
    st.markdown("### Strategy 3 Detailed Explanation")
    st.markdown(
        """
**Investment Rules :**
1. **Enter** if VIX < 20 and VVIX < 95.  
2. **Exit** if VIX > 20 or VVIX > 100.  

**Default Values (Sliders):**
- **VIX Threshold:** 20  
- **VVIX Threshold:** 95  

**Entry/Exit Details:**
- The strategy is **in market** whenever VIX is below 20 and VVIX below 95.  
- If either VIX rises above 20 **or** VVIX exceeds 100, it **exits** (goes to cash).  

**Sliders:**
- You can adjust both the **VIX Threshold** (1–40 by default) and the **VVIX Threshold** 
(1–120) to widen or narrow the conditions for entry and exit.

---
**In the App:**  
- When you pick **Strategy 1**, you’ll see a slider for **Correlation Window** (default 14 days).  
- For **Strategy 2**, you’ll see sliders for **VIX Lower/Upper**, **VVIX Lower/Upper**, and **VVIX Re-Entry Upper**.  
- For **Strategy 3**, you’ll see sliders for **VIX Threshold** and **VVIX Threshold**.  
- You can **combine** any two or three strategies with a user-specified priority, in which case the relevant parameter sliders for each chosen strategy become available.
"""
    )
