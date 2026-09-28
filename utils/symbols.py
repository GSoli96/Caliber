from utils.translations import get_text


class symbols:
    """
    Container class for UI symbols and separator options.
    
    Attributes:
        sep_options (list): List of available separator options for CSV parsing,
                           including semicolon, comma, tab, pipe, and custom option.
    """
    sep_options = [";", ",", "\\t", "|", get_text("load_dataset", "custom")]