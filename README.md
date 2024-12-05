# QuestEA

QuestEA is an exploratory research project investigating a novel approach to analyzing patient survey data by combining numerical responses with the semantic meaning of questions through embeddings. While still in early experimental stages, this academic work explores whether incorporating question semantics could potentially enhance our understanding of patient data and survey design. This is a proof-of-concept study and should not be used for clinical decisions.

## Innovative approach to extracting information from patient survey

## FAQ
### What is this?
* QuestEA (QuestionnaireEmbeddingsAnalysis) is a way to turn structured data into embeddings, initially conceived for patient surveys but applicable to many other types of data.
* The fundamental principle is that the richest and truest representation of a patient comes from combining:
    * Raw data of the patient on a metric (vectors)
    * Raw data about the metric itself (here: the meaning of each question)
* This approach is fundamentally richer than traditional methods that simply abstract questions into a list of integers (checked or not checked).

### Status?
* As I'm currently an independant researcher, I don't have access to a lot of data so I reach out to an official lab and am waiting to be granted access to official data.

### How does it work?
The process of turning patients into embeddings involves several technical steps:

1. Data Loading and Preprocessing:
   * Patient survey answers are loaded as numerical values
   * Data is normalized using either L1 or L2 normalization (configurable)
   * Survey questions are loaded as text

2. Embedding Generation:
   * For raw feature mode:
     * `feat_raw`: Uses individual answers to each question
       * Preserves the most granular information
       * Example: In a depression survey with 20 questions, creates a 20-dimensional vector
       * Each dimension represents the patient's specific answer to each question
     * `feat_agg`: Uses pre-computed aggregate scores
       * Uses established scoring methods from the survey designers
       * Example: A depression survey might aggregate 20 questions into 4 clinical subscales
       * Each dimension represents a validated clinical construct
     * Both modes can optionally apply L1/L2 normalization
   * For LLM mode (`llm_*`):
     * Survey questions are embedded using one of:
       * OpenAI's API
       * SBERT models (e.g., CLIP-ViT-B-32)
       * Custom embedding models
     * Patient answers are combined with question embeddings using element-wise multiplication
     * The result is normalized again (L1/L2)
     * Each patient ends up with a vector of the same dimension as the question embeddings

3. Dimension Reduction (Optional):
   * Can reduce dimensions using:
     * PCA (Principal Component Analysis)
     * NMF (Non-negative Matrix Factorization)
     * UMAP (Uniform Manifold Approximation and Projection)
     * Dictionary Learning
     * BVAE (Beta Variational Autoencoder)
   * Number of output dimensions is configurable

4. Final Processing:
   * Results are stored in a pandas DataFrame
   * Each row represents a patient
   * Each column represents a dimension in the embedding space
   * The embeddings can then be used for clustering, visualization, or other analyses

This process creates a rich representation that captures both:
* The patient's specific answers (through the numerical values)
* The semantic meaning of the questions (through the embeddings)

### How could this help optimize and integrate surveys?
* Survey Optimization:
    * Identifies redundant questions through correlation analysis
    * Reveals non-linear relationships between questions that only become apparent after dimensional reduction (UMAP/PaCMAP)
    * Helps create more efficient questionnaires by removing unnecessary redundancy
    * Can potentially generate new relevant questions by:
        * Finding patients who answer similarly on existing questions but differ in their symptom embeddings
        * Using [embedding-to-text techniques](https://simonwillison.net/2023/Oct/10/bottleneck/) to generate questions that maximize separation between such patients
        * While not always mathematically possible, this approach offers a novel way to discover overlooked diagnostic questions

* Better Patient Comparison:
    * Enables comparison of patients across studies with different survey combinations
    * Example: Compare three datasets where patients completed different pairs of surveys from a set of three
    * Traditional methods can't compare these patients, but QuestEA can project them into a common space

* Rich Data Integration:
    * Patient vector representation allows natural integration with:
        * Consultation summary embeddings
        * Camera feed data
        * ICD coding
        * Any other structured or unstructured patient data
    * Creates a unified way to represent and analyze diverse patient information

* Historical Data Mining:
    * Enables re-analysis of decades of historical survey data through modern embedding techniques
    * Can extract new insights from existing qualitative data like clinical observations and personality assessments
    * This vast amount of historical data could provide crucial insights into the ongoing upheaval in psychiatric classification
    * Potential to discover patterns that were impossible to detect with traditional analysis methods

### What are the main challenges in this approach?
* This is a completely new way to analyze questionnaire data that has never been done before. There is no established ground truth for patient data in this context since we're measuring it in a novel way.
* The high-dimensional nature of the data makes it particularly challenging to analyze intuitively. Human intuition breaks down when dealing with multidimensional spaces.
* Due to these challenges, we have to rely on a combination of coarse intrinsic metrics to evaluate our results.
* Ultimately, this technique needs validation with real clinical data to determine which aspects are most valuable to pursue further.

### Why did you use those specific metrics?
* Since this is a completely new approach to analyzing questionnaire data, there is no ground truth to validate against. Therefore, we rely on intrinsic metrics (like Calinski-Harabasz, Davies-Bouldin, and Silhouette scores) to evaluate the quality of the clustering results.
* These metrics help us assess how well-defined and separated the clusters are, without needing external validation data.

### What are the metrics used?
* We use several complementary clustering quality metrics:
    * Calinski-Harabasz Index (higher is better):
        * Measures ratio of between-cluster to within-cluster variance
        * Higher scores indicate better-defined clusters
        * Particularly effective for convex, well-separated clusters
    * Davies-Bouldin Index (lower is better):
        * Measures average similarity between clusters
        * Compares cluster separation vs cluster size
        * Scores closer to zero indicate better clustering
        * Limited to Euclidean space due to centroid usage
    * Silhouette Score (between -1 and +1):
        * Measures how similar points are to their own cluster vs other clusters
        * +1 indicates dense, well-separated clusters
        * 0 indicates overlapping clusters
        * -1 indicates incorrect clustering
* All these metrics work best with convex clusters and may not capture quality as well for density-based or irregularly shaped clusters

### What key questions do you hope to answer with this approach?
* The primary question is whether we can predict treatment response earlier:
    * Can we identify which patients will respond better to treatment A vs B?
    * This could help optimize treatment paths from the start
    * Could reduce time spent on ineffective treatments
    * Potentially improve patient outcomes through faster matching to effective treatments
* Other important questions include:
    * Can we identify novel patient subgroups that traditional methods miss?
    * Are there hidden patterns in historical survey data that could inform current diagnostic practices?
    * Can we better understand the relationship between different psychiatric conditions through their embedding patterns?

# How to use
### Prerequisites
1. Create a `dataset` folder in the project root
2. Download each dataset mentioned in the "List of datasets" section below by following their respective URLs
3. Place the downloaded datasets in the `dataset` folder

### Running the grid search
The `grid_search.py` script performs a comprehensive analysis across multiple parameters:

```bash
# Basic usage
python grid_search.py

# Run with custom directories
python grid_search.py --logdir=./my_logs --resultdir=./my_results

# Run in testing mode (smaller dataset)
python grid_search.py --testing

# Enable debug mode
python grid_search.py --debug

# Enable verbose output
python grid_search.py --verbose
```

The script will:
1. Process each dataset through multiple embedding methods
2. Apply various dimensionality reduction techniques
3. Perform clustering with different methods
4. Generate visualizations and metrics
5. Save results in the specified directories

View results in tensorboard:
```bash
tensorboard --logdir=./tensorboard_runs
```

# Notes
* Many docstrings in this project were initially generated using [aider](https://aider.chat/) and were quickly reviewed for validity. The clarity of the code is also somewhat below my standards but should be okay.
- No requirements.txt file is provided, you are expected to install the needed package as you get ImportError, PR welcome

# List of datasets used

### openpsychometrics data

<details><summary>More</summary>

* example publications with this data: https://openpsychometrics.org/_rawdata/cited/
* source : https://openpsychometrics.org/_rawdata/
    * kaggle link : https://www.kaggle.com/code/essadeqelaamiri/personality-k-means

</details>


### 16PF

<details><summary>More</summary>

* from openpsychometrics
* n=49,159
* Answers to Cattell's 16 Personality Factors Test with items from the IPIP.
* 163 likert rated items, gender, age, country and accuracy.

</details>


### Adolescent depression data: Social Reward Questionnaire Adolescent

<details><summary>More</summary>

* https://datadryad.org/stash/dataset/doi:10.5061%2Fdryad.n399g
* SRQA codes found at https://osf.io/9w6yq then saved as pdf and parsed as txt
* n=568, 20 items
* Abstract
    * During adolescence, social interactions are a potent source of reward. However, no measure of social reward value exists for this age group. In this study, we adapted the adult Social Reward Questionnaire, which we had previously developed and validated, for use with adolescents. Participants aged 11–16 (n = 568; 50% male) completed the Social Reward Questionnaire—Adolescent Version (SRQ-A), alongside measures of personality traits—five-factor model (FFM) and callous–unemotional (CU) traits—for construct validity purposes. A confirmatory factor analysis of the SRQ-A supported a five-factor structure (Comparative Fit Index = 0.90; Root Mean Square Error of Approximation = 0.07), equating to five questionnaire subscales: enjoyment of Admiration, Negative Social Potency, Passivity, Prosocial Interactions and Sociability. Associations with FFM and CU traits were in line with what is seen for adult samples, providing support for the meaning of SRQ-A subscales in adolescents. In particular, adolescents with high levels of CU traits showed an ‘inverted’ pattern of social reward, in which being cruel is enjoyable and being kind is not. Gender invariance was also assessed and was partially supported. The SRQ-A is a valid, reliable measure of individual differences in social reward in adolescents. 

</details>


### DASS

<details><summary>More</summary>

* subcales from :
    * https://neurocogsystem.com/wp-content/uploads/2021/02/DASS-42-Scoring.pdf
* n=39,775; 42 DASS items, 30+ personality and demographic items

</details>


### hamilton depression scale

<details><summary>More</summary>

* dataset link : http://d-scholarship.pitt.edu/35396/
    * https://datacatalog.hsls.pitt.edu/dataset/29
* potential other hamilton source:
    * https://datasetsearch.research.google.com/search?src=0&query=hamilton%20depression&docid=L2cvMTFqbnoxOXlibg%3D%3D

</details>


### HEXACO

<details><summary>More</summary>

* from openpsychometrics
* n = 22,786
* 240 scale rated items, country

</details>


### IPIP

<details><summary>More</summary>

* from openpsychometrics
* big 5 data
* 50 items, and technical information
* n=1,015,342

</details>


### Robb Rutledge smartphone dataset

<details><summary>More</summary>

* https://datadryad.org/stash/dataset/doi:10.5061/dryad.prr4xgxkk
* BDI : https://arc.psych.wisc.edu/self-report/beck-depression-inventory-bdi/
* Actually the BDI questions are formatted such that you answer a number that corresponds to a sentence. So it's a good way to test the robustness of the embeddings.
* n=1,858
* Abstract
    * This resource consists of data from a risky decision and happiness task that was part of The Great Brain Experiment (GBE) smartphone app. Data were collected from 47,067 participants aged 18+ between March 8, 2013 and October 5, 2015. These anonymous unpaid participants completed the task a total of 91,058 times making approximately 2.7 million choices and 1.1 million happiness ratings in total. This resource represents at least 6,000 hours of task data. A subset of 1,858 participants also completed a depression questionnaire and answered five questions about their depression history.

</details>


# Notes about the intrinsic metrics used
* source [scikit-learn documentation on clustering metrics](https://scikit-learn.org/stable/modules/clustering.html)
* Calinski-Harabasz Index:
    * also known as the Variance Ratio Criterion - can be used to evaluate the model, where a higher Calinski-Harabasz score relates to a model with better defined clusters.
        * The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
        * The Calinski-Harabasz index is generally higher for convex clusters than other concepts of clusters, such as density based clusters like those obtained through DBSCAN.
* Davies-Bouldin Index
    * a lower Davies-Bouldin index relates to a model with better separation between the clusters.
    * This index signifies the average ‘similarity’ between clusters, where the similarity is a measure that compares the distance between clusters with the size of the clusters themselves.
    * Zero is the lowest possible score. Values closer to zero indicate a better partition.
    * The Davies-Boulding index is generally higher for convex clusters than other concepts of clusters, such as density based clusters like those obtained from DBSCAN.
    * The usage of centroid distance limits the distance metric to Euclidean space.
* Silhouette Score
    * The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.
    * The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    * The Silhouette Coefficient is generally higher for convex clusters than other concepts of clusters, such as density based clusters like those obtained through DBSCAN.


# Preliminary Results

⚠️ **WARNING: HIGHLY PRELIMINARY RESULTS** ⚠️

The following results should be interpreted with extreme caution:
* Based on publicly available datasets of questionable quality
* May not be reproducible with clinical-grade data
* Primarily serves as a proof-of-concept
* Should not be used for any clinical decisions

Key findings from our **initial** experiments:

* Interesting results for the Rutledge BDI dataset:
    * Almost all models perform well with aggregated features, but not with raw features
    * This is particularly noteworthy because the models ultimately take raw features as input
    * The fact that results correlate well with aggregated features is remarkable, especially since the aggregated features themselves don't correlate strongly with raw features
    * Note: Data preprocessing included projection onto a circle

# Roadmap / TODOs
<!-- BEGIN_TODO -->
- add pacmap as a dim reduction
    - https://github.com/YingfanWang/PaCMAP
- use a metric that takes monotony into account
    - if the function is not monotonic then there's a peak that shows a better clustering on a given metric
        - especially curious for the results on Big5 data
- See if you can get your hands on STAR*D data, HBNN data, UKB data
- In the random vector test, add a test with one hot vectors and few hot vectors
<!-- END_TODO -->
