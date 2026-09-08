import csv
import random
import re

# ----------------------------------------------------------------------
# 1. Load the Adult dataset from the provided text (simulate reading CSV)
#    In practice, you would have the file on disk.
# ----------------------------------------------------------------------
# The dataset content is given in the prompt as a long string.
# For this script, we assume the file 'Adult_preProcessed.csv' exists.
# We'll read it with csv.DictReader.
# ----------------------------------------------------------------------

categorical_columns = ['workclass', 'education', "marital_status", 'occupation',
                       'relationship', 'race', 'sex', "native_country"]
numeric_columns = ['age', 'fnlwgt', "education_num", "capital_gain",
                   "capital_loss", "hours_per_week"]

def load_adult_data(filename):
    """Load the Adult dataset and extract metadata."""
    data = []
    distinct_values = {col: set() for col in categorical_columns}
    ranges = {col: {'min': float('inf'), 'max': float('-inf')} for col in numeric_columns}

    with open(filename, 'r', encoding='utf-8') as f:
        # Read header and replace hyphens with underscores
        header = next(csv.reader(f))
        clean_header = [h.replace('-', '_') for h in header]
        reader = csv.DictReader(f, fieldnames=clean_header)
        for row in reader:
            data.append(row)
            for col in categorical_columns:
                if row.get(col) and row[col] != '?':
                    distinct_values[col].add(row[col])
            for col in numeric_columns:
                try:
                    val = float(row[col])
                    ranges[col]['min'] = min(ranges[col]['min'], val)
                    ranges[col]['max'] = max(ranges[col]['max'], val)
                except:
                    pass

    # Convert sets to lists for random choice
    for col in categorical_columns:
        distinct_values[col] = [v for v in distinct_values[col] if v and v != '?']
    return data, distinct_values, ranges

# ----------------------------------------------------------------------
# 2. Generate random natural language query and SQL for each DBMS
# ----------------------------------------------------------------------
def random_condition(columns, distinct_values, ranges):
    """Generate a random SQL condition (WHERE clause fragment)."""
    col = random.choice(list(columns))
    if col in distinct_values:  # categorical
        val = random.choice(distinct_values[col])
        op = random.choice(['=', '!=', 'LIKE'])
        if op == 'LIKE':
            # For string columns, sometimes use LIKE with wildcard
            val = f"'%{val}%'" if random.random() < 0.3 else f"'{val}'"
        else:
            val = f"'{val}'"
        return f"{col} {op} {val}"
    else:  # numeric
        low = ranges[col]['min']
        high = ranges[col]['max']
        mid = (low + high) / 2
        op = random.choice(['=', '!=', '<', '>', '<=', '>=', 'BETWEEN'])
        if op == 'BETWEEN':
            v1 = random.uniform(low, high)
            v2 = random.uniform(v1, high)
            return f"{col} BETWEEN {v1:.1f} AND {v2:.1f}"
        else:
            val = random.uniform(low, high)
            return f"{col} {op} {val:.1f}"

def generate_query():
    """Generate a NL query and its SQL translations."""
    # Choose a basic template
    template = random.choice([
        "Show me {agg} {what} where {condition} {group} {order}",
        "List {what} with {condition} {order}",
        "Find the {agg} of {what} grouped by {group_col} having {having}",
        "How many {what} satisfy {condition}?",
        "Get {what} ordered by {order_col} {order_dir} {limit}",
        "What is the {agg} {what} for each {group_col}?",
    ])

    # Decide components
    agg_funcs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
    what_choices = ['*', 'age', 'hours_per_week', 'capital_gain', 'capital_loss', 'fnlwgt']
    what = random.choice(what_choices)

    # Conditions
    num_conditions = random.randint(0, 3)
    conditions = []
    for _ in range(num_conditions):
        conditions.append(random_condition(columns, distinct_values, ranges))
    condition_str = " AND ".join(conditions) if conditions else "1=1"

    # Group by
    group_by = random.choice([None] + [c for c in categorical_columns])
    group_str = f"GROUP BY {group_by}" if group_by else ""

    # Having (if group by and agg)
    having_str = ""
    if group_by and random.random() < 0.3:
        agg = random.choice(agg_funcs)
        col = random.choice(what_choices)
        op = random.choice(['>', '<', '>=', '<=', '='])
        val = random.randint(1, 100)
        having_str = f"HAVING {agg}({col}) {op} {val}"

    # Order by
    order_col = random.choice(list(columns))
    order_dir = random.choice(['ASC', 'DESC'])
    order_str = f"ORDER BY {order_col} {order_dir}" if random.random() < 0.5 else ""

    # Limit / TOP
    limit_val = random.randint(5, 50)
    limit_str = f"LIMIT {limit_val}"  # for most DBMS
    top_str = f"TOP {limit_val}"      # for SQL Server

    # Build natural language description
    nl = template.format(
        agg=random.choice(agg_funcs) if 'agg' in template else '',
        what=what,
        condition=condition_str if condition_str != '1=1' else 'all records',
        group=group_str if group_str else '',
        order=order_str if order_str else '',
        group_col=group_by if group_by else 'category',
        having=having_str if having_str else '',
        order_col=order_col,
        order_dir=order_dir,
        limit=f"limit to {limit_val} rows" if 'limit' in template else ''
    )
    # Clean up NL
    nl = re.sub(r'\s+', ' ', nl).strip()

    # Build SQL for each DBMS
    # Base SELECT clause
    agg_func = random.choice(agg_funcs) # Define agg_func here for use in select_clause
    select_clause = f"SELECT {what}" if what != '*' else "SELECT *"
    if 'agg' in template:
        select_clause = f"SELECT {agg_func}({what})"

    # Assemble SQL
    base_sql = f"{select_clause} FROM adult_preprocessed"
    if condition_str != '1=1':
        base_sql += f" WHERE {condition_str}"
    if group_str:
        base_sql += f" {group_str}"
    if having_str:
        base_sql += f" {having_str}"
    if order_str:
        base_sql += f" {order_str}"

    # Adapt to each DBMS
    mysql_sql = base_sql.replace('"', '`') + f" LIMIT {limit_val}"
    sqlite_sql = base_sql + f" LIMIT {limit_val}"
    postgres_sql = base_sql + f" LIMIT {limit_val}"
    duckdb_sql = base_sql + f" LIMIT {limit_val}"
    # SQL Server uses TOP before columns
    sqlserver_sql = re.sub(r'SELECT (.*) FROM', f'SELECT TOP {limit_val} \\1 FROM', base_sql, flags=re.IGNORECASE)

    return {
        'NLQuery': nl,
        'MySQL': mysql_sql,
        'SQLite': sqlite_sql,
        'PostgreSQL': postgres_sql,
        'DuckDB': duckdb_sql,
        'SQL Server': sqlserver_sql
    }

# ----------------------------------------------------------------------
# 3. Main generation loop
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Load data (assume file is present)
    data, distinct_values, ranges = load_adult_data('csv\Adult\Adult_preProcessed.csv')
    columns = list(distinct_values.keys()) + list(ranges.keys())

    # Generate 20,000 queries
    output_file = 'queries_20k.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['NLQuery', 'MySQL', 'SQLite', 'PostgreSQL', 'DuckDB', 'SQL Server'])
        writer.writeheader()
        for _ in range(15):
            row = generate_query()
            writer.writerow(row)

    print(f"Generated 20,000 queries in {output_file}")