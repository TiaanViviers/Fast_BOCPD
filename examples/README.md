# Fast BOCPD Examples & Tutorials

Learn how to use Fast BOCPD through interactive Jupyter notebooks.

## Tutorial Sequence

Work through these in order for the best learning experience:

### 1. **Quickstart** (`01_quickstart.ipynb`) ⚡
**Time:** 5 minutes  
**You'll learn:**
- Basic changepoint detection
- Batch vs online processing
- How to visualize results

**Start here!** This gets you up and running quickly.

---

### 2. **Online vs Batch** (`02_online_vs_batch.ipynb`)
**Time:** 10 minutes  
**You'll learn:**
- When to use online mode (streaming data)
- When to use batch mode (historical data)
- Performance differences
- Using `OnlineChangeDetector` utility

**Best for:** Understanding which mode fits your use case

---

### 3. **Understanding Outputs** (`03_understanding_outputs.ipynb`) 
**Time:** 15 minutes  
**You'll learn:**
- What is `cp_prob` (changepoint probability)?
- What is `posterior_r` (run length distribution)?
- What is MAP run length?
- How to use confidence scores
- Visualizing uncertainty

**Best for:** Deep understanding of what BOCPD tells you

---

### 4. **Parameter Guide** (`04_parameter_guide.ipynb`)
**Time:** 15 minutes  
**You'll learn:**
- How to choose observation models
  - GaussianNIG: When and why?
  - What do hyperparameters mean?
- How to set hazard functions
  - What is λ (lambda)?
  - How to estimate expected run length
- Tuning `max_run_length`
- Visual parameter sensitivity analysis

**Best for:** Configuring BOCPD for your specific data

---

### 5. **Advanced Features** (`05_advanced_features.ipynb`)
**Time:** 10 minutes  
**You'll learn:**
- `OnlineChangeDetector` for production
- Segment extraction
- MAP confidence scores
- State management (reset, persistence)
- Handling edge cases

**Best for:** Building production systems

---

### 6. **Real-World Example: US30 Volatility** (`06_real_world_us30_volatility.ipynb`)
**Time:** 20 minutes  
**You'll learn:**
- Complete workflow with real financial data
- Data preprocessing
- Detecting volatility regime changes
- Interpreting results
- Production-ready code template

**Best for:** See how it works on real data

---

## 🚀 Quick Start

```bash
# Install Fast BOCPD
pip install -e ..

# Install notebook dependencies
pip install jupyter matplotlib

# Launch Jupyter
jupyter notebook
```

Then open `01_quickstart.ipynb` and start learning!

---

## 💡 Tips

- **New to BOCPD?** Start with #1 (Quickstart)
- **Want to understand the math?** Focus on #3 (Understanding Outputs)
- **Building a real system?** Skip to #5 (Advanced) and #6 (Real-World)
- **Have your own data?** Follow #6 as a template

---

## 📖 Additional Resources

- **Documentation:** See `../README.md`
- **Tests:** See `../tests/` for more usage examples
- **API Reference:** See docstrings in `../fast_bocpd/`

Happy changepoint detecting! 🎯
