# Synthetic Data Generation - Installation Guide

## Required Libraries

The synthetic data generation tab requires the following Python libraries:

```bash
# Core library for simple data generation
pip install faker

# Advanced synthetic data generation
pip install sdv

# For Excel export functionality
pip install openpyxl
```

## Installation Commands

### Install All Dependencies

```bash
pip install faker sdv openpyxl
```

### Individual Installation

If you only want specific strategies:

**For Faker only:**
```bash
pip install faker
```

**For SDV strategies (GaussianCopula, CTGAN, TVAE, HMA1):**
```bash
pip install sdv
```

**For Excel export:**
```bash
pip install openpyxl
```

## Verification

After installation, verify the libraries are available:

```bash
python -c "import faker; print('Faker:', faker.__version__)"
python -c "import sdv; print('SDV:', sdv.__version__)"
python -c "import openpyxl; print('OpenPyXL:', openpyxl.__version__)"
```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Navigate to the "🧬 Synthetic Data" tab

3. Select a generation strategy:
   - **Faker**: Simple realistic data (no training required)
   - **GaussianCopula**: Statistical modeling (requires sample data)
   - **CTGAN**: Deep learning GAN (requires sample data)
   - **TVAE**: Deep learning VAE (requires sample data)
   - **HMA1**: Multi-table relational (advanced)

4. Configure parameters and generate data

5. View analytics, download, or create database

## Troubleshooting

### Library Not Found

If you see "library not installed" errors:
1. Check that you're using the correct Python environment
2. Reinstall the library: `pip install --upgrade <library_name>`
3. Restart the Streamlit application

### Import Errors

If you see import errors when running the app:
1. Ensure all dependencies are installed
2. Check Python version compatibility (Python 3.8+ recommended)
3. Try: `pip install --upgrade sdv faker openpyxl`

### Performance Issues

For CTGAN/TVAE:
- Start with small datasets (< 10,000 rows)
- Use fewer epochs (50-100) for testing
- Increase epochs (300-500) for production quality
- Consider using GPU if available

## Optional: Virtual Environment

Recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install faker sdv openpyxl

# Run application
streamlit run app.py
```
