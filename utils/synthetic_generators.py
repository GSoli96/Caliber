"""
Synthetic Data Generation Utilities

This module provides functions to generate synthetic datasets using various libraries:
- Faker: Simple realistic tabular data
- SDV (Synthetic Data Vault): Advanced statistical and deep learning approaches
  - GaussianCopula: Numerical data with statistical properties
  - RelationalSynthesizer: Multi-table relational data
  - CTGAN/TVAE: Deep learning for complex datasets
  - HMA1: Hierarchical multi-table modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings

# ============================================================================
# LIBRARY AVAILABILITY CHECKS
# ============================================================================

def check_library_available(library_name: str) -> Tuple[bool, Optional[str]]:
    """Check if a library is available and return import error if not."""
    try:
        if library_name == "faker":
            import faker
            return True, None
        elif library_name == "sdv":
            import sdv
            return True, None
        elif library_name == "sdv.single_table":
            from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
            return True, None
        elif library_name == "sdv.multi_table":
            from sdv.multi_table import HMASynthesizer
            return True, None
        else:
            return False, f"Unknown library: {library_name}"
    except ImportError as e:
        return False, str(e)


# ============================================================================
# FAKER STRATEGY
# ============================================================================

def generate_faker_data(
    num_rows: int,
    columns: List[Tuple[str, str]],
    locale: str = 'en_US',
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> pd.DataFrame:
    """
    Generate synthetic data using Faker library.
    
    Parameters:
    -----------
    num_rows : int
        Number of rows to generate
    columns : List[Tuple[str, str]]
        List of (column_name, faker_provider) tuples
        Examples: ('name', 'name'), ('email', 'email'), ('address', 'address')
    locale : str
        Faker locale (default: 'en_US')
    seed : Optional[int]
        Random seed for reproducibility
    progress_callback : Optional[Callable[[int, int], None]]
        Callback function(current, total) for progress updates
        
    Returns:
    --------
    pd.DataFrame
        Generated synthetic dataset
        
    Raises:
    -------
    ImportError
        If faker library is not installed
    ValueError
        If invalid parameters are provided
    """
    available, error = check_library_available("faker")
    if not available:
        raise ImportError(
            f"Faker library is not installed. Error: {error}\n"
            "Install with: pip install faker"
        )
    
    from faker import Faker
    
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    if not columns:
        raise ValueError("columns list cannot be empty")
    
    fake = Faker(locale)
    if seed is not None:
        Faker.seed(seed)
        np.random.seed(seed)
    
    data = {}
    for col_name, provider in columns:
        data[col_name] = []
        
        # Get the faker method
        try:
            faker_method = getattr(fake, provider)
        except AttributeError:
            raise ValueError(f"Invalid Faker provider: {provider}")
        
        # Generate data
        for i in range(num_rows):
            try:
                data[col_name].append(faker_method())
            except Exception as e:
                # Fallback to string representation if provider fails
                data[col_name].append(f"Error: {str(e)}")
            
            if progress_callback and i % 100 == 0:
                progress_callback(i, num_rows)
    
    if progress_callback:
        progress_callback(num_rows, num_rows)
    
    return pd.DataFrame(data)


# ============================================================================
# SDV - GAUSSIAN COPULA STRATEGY
# ============================================================================

def generate_gaussian_copula_data(
    metadata: Optional[Dict] = None,
    real_data: Optional[pd.DataFrame] = None,
    num_rows: int = 1000,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> pd.DataFrame:
    """
    Generate synthetic data using SDV GaussianCopula synthesizer.
    
    Parameters:
    -----------
    metadata : Optional[Dict]
        SDV metadata dictionary defining the schema
    real_data : Optional[pd.DataFrame]
        Real data to learn from (required if metadata is not provided)
    num_rows : int
        Number of synthetic rows to generate
    seed : Optional[int]
        Random seed for reproducibility
    progress_callback : Optional[Callable[[str], None]]
        Callback function(message) for progress updates
        
    Returns:
    --------
    pd.DataFrame
        Generated synthetic dataset
        
    Raises:
    -------
    ImportError
        If SDV library is not installed
    ValueError
        If neither metadata nor real_data is provided
    """
    available, error = check_library_available("sdv.single_table")
    if not available:
        raise ImportError(
            f"SDV library is not installed. Error: {error}\n"
            "Install with: pip install sdv"
        )
    
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    
    if real_data is None and metadata is None:
        raise ValueError("Either real_data or metadata must be provided")
    
    if progress_callback:
        progress_callback("Initializing GaussianCopula synthesizer...")
    
    # Create synthesizer
    if metadata:
        synthesizer = GaussianCopulaSynthesizer(metadata)
    else:
        # Auto-detect metadata from real data
        metadata_obj = SingleTableMetadata()
        metadata_obj.detect_from_dataframe(real_data)
        synthesizer = GaussianCopulaSynthesizer(metadata_obj)
    
    # Fit to real data if provided
    if real_data is not None:
        if progress_callback:
            progress_callback("Fitting model to real data...")
        synthesizer.fit(real_data)
    
    # Generate synthetic data
    if progress_callback:
        progress_callback(f"Generating {num_rows} synthetic rows...")
    
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    
    if progress_callback:
        progress_callback("Generation complete!")
    
    return synthetic_data


# ============================================================================
# SDV - CTGAN STRATEGY
# ============================================================================

def generate_ctgan_data(
    real_data: pd.DataFrame,
    num_rows: int = 1000,
    epochs: int = 300,
    batch_size: int = 500,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> pd.DataFrame:
    """
    Generate synthetic data using SDV CTGAN (Conditional Tabular GAN).
    
    CTGAN uses deep learning (GANs) to generate high-quality synthetic data.
    Best for complex, high-dimensional datasets with mixed data types.
    
    Parameters:
    -----------
    real_data : pd.DataFrame
        Real data to learn from
    num_rows : int
        Number of synthetic rows to generate
    epochs : int
        Number of training epochs (more = better quality, slower)
    batch_size : int
        Training batch size
    seed : Optional[int]
        Random seed for reproducibility
    progress_callback : Optional[Callable[[str], None]]
        Callback function(message) for progress updates
        
    Returns:
    --------
    pd.DataFrame
        Generated synthetic dataset
        
    Raises:
    -------
    ImportError
        If SDV library is not installed
    """
    available, error = check_library_available("sdv.single_table")
    if not available:
        raise ImportError(
            f"SDV library is not installed. Error: {error}\n"
            "Install with: pip install sdv"
        )
    
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    
    if progress_callback:
        progress_callback("Initializing CTGAN synthesizer...")
    
    # Auto-detect metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    # Create synthesizer
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    # Fit to real data
    if progress_callback:
        progress_callback(f"Training CTGAN model ({epochs} epochs)...")
    
    synthesizer.fit(real_data)
    
    # Generate synthetic data
    if progress_callback:
        progress_callback(f"Generating {num_rows} synthetic rows...")
    
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    
    if progress_callback:
        progress_callback("Generation complete!")
    
    return synthetic_data


# ============================================================================
# SDV - TVAE STRATEGY
# ============================================================================

def generate_tvae_data(
    real_data: pd.DataFrame,
    num_rows: int = 1000,
    epochs: int = 300,
    batch_size: int = 500,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> pd.DataFrame:
    """
    Generate synthetic data using SDV TVAE (Tabular Variational AutoEncoder).
    
    TVAE uses deep learning (VAE) to generate synthetic data.
    Generally faster than CTGAN and works well with mixed data types.
    
    Parameters:
    -----------
    real_data : pd.DataFrame
        Real data to learn from
    num_rows : int
        Number of synthetic rows to generate
    epochs : int
        Number of training epochs (more = better quality, slower)
    batch_size : int
        Training batch size
    seed : Optional[int]
        Random seed for reproducibility
    progress_callback : Optional[Callable[[str], None]]
        Callback function(message) for progress updates
        
    Returns:
    --------
    pd.DataFrame
        Generated synthetic dataset
        
    Raises:
    -------
    ImportError
        If SDV library is not installed
    """
    available, error = check_library_available("sdv.single_table")
    if not available:
        raise ImportError(
            f"SDV library is not installed. Error: {error}\n"
            "Install with: pip install sdv"
        )
    
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    
    if progress_callback:
        progress_callback("Initializing TVAE synthesizer...")
    
    # Auto-detect metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    # Create synthesizer
    synthesizer = TVAESynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Fit to real data
    if progress_callback:
        progress_callback(f"Training TVAE model ({epochs} epochs)...")
    
    synthesizer.fit(real_data)
    
    # Generate synthetic data
    if progress_callback:
        progress_callback(f"Generating {num_rows} synthetic rows...")
    
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    
    if progress_callback:
        progress_callback("Generation complete!")
    
    return synthetic_data


# ============================================================================
# SDV - HMA1 (Hierarchical Multi-table) STRATEGY
# ============================================================================

def generate_hma1_data(
    real_data: Dict[str, pd.DataFrame],
    metadata: Dict,
    num_rows: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic relational data using SDV HMA1 synthesizer.
    
    HMA1 (Hierarchical Modeling Algorithm) preserves relationships between tables
    including primary/foreign keys and referential integrity.
    
    Parameters:
    -----------
    real_data : Dict[str, pd.DataFrame]
        Dictionary of table_name -> DataFrame with real data
    metadata : Dict
        SDV multi-table metadata defining relationships
    num_rows : Optional[int]
        Number of rows for root table (None = same as original)
    progress_callback : Optional[Callable[[str], None]]
        Callback function(message) for progress updates
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of table_name -> synthetic DataFrame
        
    Raises:
    -------
    ImportError
        If SDV library is not installed
    """
    available, error = check_library_available("sdv.multi_table")
    if not available:
        raise ImportError(
            f"SDV multi-table library is not installed. Error: {error}\n"
            "Install with: pip install sdv"
        )
    
    from sdv.multi_table import HMASynthesizer
    
    if progress_callback:
        progress_callback("Initializing HMA synthesizer...")
    
    # Create synthesizer
    synthesizer = HMASynthesizer(metadata)
    
    # Fit to real data
    if progress_callback:
        progress_callback("Fitting model to relational data...")
    
    synthesizer.fit(real_data)
    
    # Generate synthetic data
    if progress_callback:
        msg = f"Generating synthetic relational data..."
        if num_rows:
            msg = f"Generating {num_rows} rows for root table..."
        progress_callback(msg)
    
    if num_rows:
        synthetic_data = synthesizer.sample(scale=num_rows / len(list(real_data.values())[0]))
    else:
        synthetic_data = synthesizer.sample()
    
    if progress_callback:
        progress_callback("Generation complete!")
    
    return synthetic_data


# ============================================================================
# COMMON FAKER PROVIDERS
# ============================================================================

COMMON_FAKER_PROVIDERS = {
    "Personal": [
        ("name", "Full Name"),
        ("first_name", "First Name"),
        ("last_name", "Last Name"),
        ("prefix", "Prefix (Mr., Mrs.)"),
        ("suffix", "Suffix (Jr., Sr.)"),
    ],
    "Contact": [
        ("email", "Email Address"),
        ("phone_number", "Phone Number"),
        ("address", "Full Address"),
        ("street_address", "Street Address"),
        ("city", "City"),
        ("state", "State"),
        ("zipcode", "ZIP Code"),
        ("country", "Country"),
    ],
    "Internet": [
        ("url", "URL"),
        ("domain_name", "Domain Name"),
        ("ipv4", "IPv4 Address"),
        ("ipv6", "IPv6 Address"),
        ("mac_address", "MAC Address"),
        ("user_name", "Username"),
    ],
    "Business": [
        ("company", "Company Name"),
        ("job", "Job Title"),
        ("catch_phrase", "Company Slogan"),
        ("bs", "Business Speak"),
    ],
    "Financial": [
        ("credit_card_number", "Credit Card Number"),
        ("credit_card_provider", "Credit Card Provider"),
        ("iban", "IBAN"),
        ("bban", "BBAN"),
    ],
    "Date/Time": [
        ("date", "Date"),
        ("date_time", "Date Time"),
        ("time", "Time"),
        ("year", "Year"),
    ],
    "Text": [
        ("text", "Random Text"),
        ("sentence", "Sentence"),
        ("paragraph", "Paragraph"),
        ("word", "Word"),
    ],
    "Numeric": [
        ("random_int", "Random Integer"),
        ("random_number", "Random Number"),
        ("pyfloat", "Random Float"),
    ],
}
