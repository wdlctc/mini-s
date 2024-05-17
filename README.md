
## Installation

To install and run the mini-s model, follow these steps:

1. Clone the repository:

   ```
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Navigate to the `transformers` directory:

   ```
   cd transformers/
   ```

4. Install the `transformers` package in editable mode:

   ```
   pip install -e .
   ```

5. Install the `flash-attn` package:

   ```
   pip install flash-attn --no-build-isolation
   ```

   Note: The `--no-build-isolation` flag is used to avoid potential build conflicts.

## Usage

To run the benchmark script and evaluate the performance of the mini-s model, use the following command:

```
python benchmark.py
```
