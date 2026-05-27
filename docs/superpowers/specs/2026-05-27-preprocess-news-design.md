# Design Spec: Preprocessing Vietnamese News Data using TF-IDF

## Objective
Convert raw Vietnamese news documents classified into 10 topics into numeric vectors using the TF-IDF representation, while properly handling Vietnamese word segmentation and stopwords.

## Proposed Design

### 1. Stopwords Alignment
- **File**: `2-Preprocessing/Assignment/News_VNExpress_/vietnamese-stopwords.txt`
- **Processing**:
  - Read lines from the file.
  - Remove leading/trailing whitespaces.
  - Replace spaces within phrases with underscores `_` to match the output format of `ViTokenizer` (e.g., `bao giờ` -> `bao_giờ`).
  - Store as a unique list/set of stopwords.

### 2. Document Tokenization
- **Source**: `data_train.data`
- **Processing**:
  - Iterate through all text documents in the dataset.
  - Apply `pyvi.ViTokenizer.tokenize(doc)` to segment Vietnamese compound words.
  - Keep the output as a list of space-separated tokenized strings.

### 3. Numerical Vectorization (TF-IDF)
- **BoW Representation**:
  - Instantiate `CountVectorizer(stop_words=stopwords)` using the aligned stopwords list.
  - Fit and transform tokenized documents using the vectorizer.
- **TF-IDF Transformation**:
  - Instantiate `TfidfTransformer()`.
  - Fit and transform the bag-of-words matrix to compute TF-IDF values.
  - Assign to `data_preprocessed`.

## Verification Plan
- Verify that `module_count_vector` contains the correct vocabulary.
- Verify the shape of $X$ is $(1339, m)$ where $m$ is the vocabulary size.
- Verify the shape of $Y$ is $(1339,)$.
- Print out sample sparse matrix representation for a single document to ensure it processed successfully.
