# Twitter-Sentiment-Extraction-Analysis-and-performing-EDA
## Project Overview

This project focuses on analyzing and visualizing a dataset containing social media user data using VADER Sentiment Analysis tool . The goal is to explore user trends, follower statistics, and hashtags, leveraging interactive visualizations and Natural Language Processing (NLP) techniques. By performing comprehensive data preprocessing, exploratory data analysis (EDA), and hashtag cleaning, this project lays the groundwork for future modeling tasks like sentiment analysis or user behavior prediction.

Key highlights of this project include:

Cleaning and transforming complex datasets with mixed data types.
Using Plotly for dynamic, interactive visualizations of user metrics.
Conducting NLP-based hashtag analysis and preparing data for advanced text modeling.
This repository serves as a showcase of data preprocessing, EDA, and visual storytelling, offering insights into both technical and creative aspects of data analysis.
## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset Highlights](#dataset-highlights)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [User Trends](#user-trends)
  - [Follower Statistics](#follower-statistics)
  - [Hashtag Analysis](#hashtag-analysis)
- [Visual Storytelling](#visual-storytelling)
- [How to Use](#how-to-use)
- [Conclusion](#conclusion)
- [References](#references)

## Technologies Used
- Python (Numpy, Pandas, Matplotlib, Seaborn)
- Plotly (Interactive Visualizations)
- NLTK (Text Processing and Sentiment Analysis)
- VADER Sentiment Analysis Tool
- WordCloud (Text Visualization)
- TQDM (Progress Tracking)

## Dataset Highlights
The dataset (tweets.csv) used in this project contains a rich set of attributes including user information (user_name, user_location, user_description), tweet content (text, hashtags), and metadata (date, source). The dataset may have missing values, which are handled carefully during the analysis process to ensure robust insights.

[Dataset link]()

``` df.info() ```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 725835 entries, 0 to 725834
Data columns (total 12 columns):
 #   Column            Non-Null Count   Dtype 
---  ------            --------------   ----- 
 0   user_name         674488 non-null  object
 1   text              373448 non-null  object
 2   user_location     281894 non-null  object
 3   user_description  346450 non-null  object
 4   user_created      307134 non-null  object
 5   user_followers    307134 non-null  object
 6   user_friends      307134 non-null  object
 7   user_favourites   307134 non-null  object
 8   user_verified     307131 non-null  object
 9   date              307127 non-null  object
 10  hashtags          242094 non-null  object
 11  source            307097 non-null  object
dtypes: object(12)
memory usage: 66.5+ MB
```
![image](https://github.com/user-attachments/assets/565204eb-744e-45a1-a120-41fd20fe2ecf)


## Data Preprocessing
- Handling missing values
  This dataset contains lot of null values almost half of the dataset in some features so dropping all of them will reduce our dataset size and it will be very small which can degrade our analysis quality .
  So we are dropping only those row where all important features' values are null.
  ![image](https://github.com/user-attachments/assets/abb70859-8950-4acd-b728-4680c7b94a1c)
  
  But here problem is that these kind of data is also very big, so we can't just drop all of them:
  ![image](https://github.com/user-attachments/assets/3c24ef3a-597b-4deb-b187-ab01136be349)

 As you can see in the username column there are some hashtags, we can extract these hashtags and fill the corresponding null values in hashtags column.
 ```
  # Function to extract hashtags from text
def extract_hashtags(text):
    if pd.notna(text):  # Check if text is not null
        return " ".join(re.findall(r"#\w+", text))
    return ""

# Apply the function to the 'user_name' column
df['extracted_hashtags'] = df['user_name'].apply(extract_hashtags)

 ```

```
  # Fill missing 'hashtags' with 'extracted_hashtags' if available
df['hashtags'] = df.apply(
    lambda row: row['extracted_hashtags'] if pd.isna(row['hashtags']) else row['hashtags'], axis=1
)
```
Now after filling all possible nan values of hashtags column, let us check still there are some columns having missing values in all these columns 'text', 'hashtags', and 'user_description':
![image](https://github.com/user-attachments/assets/61059f96-0939-4b8b-9a7f-90a5acc47763)

As we can now there are around 76k row of this kind which is very less in compare 340k, so we will drop these data:
```
#drop all the rows which are in df_thu
df.drop(df_thu.index, inplace=True)
```

Drop errelevant features like 'user_created','user_favourites','user_friends','source','user_verified':
```
df.drop(columns=['user_created','user_favourites','user_friends','source','user_verified'], inplace=True)
```

In hashtags column there were inconsistent datatypes, some in python list which was good for us but others were in string and some values were null. 
So we need to convert all of them in python lists to perform further operations.

```
import ast

def clean_hashtags(entry):
    try:
        # Parse the string into a Python object (handles strings like "['tag1', 'tag2']")
        if isinstance(entry, str):
            entry = ast.literal_eval(entry)

        # Flatten nested lists, if any
        if isinstance(entry, list):
            flattened = []
            for item in entry:
                if isinstance(item, list):
                    flattened.extend(item)  # Unwrap nested lists
                else:
                    flattened.append(item)

            # Remove stray commas and empty entries
            cleaned = [str(tag).strip() for tag in flattened if tag and tag != ","]
            return cleaned
    except (ValueError, SyntaxError):
        # If parsing fails, return an empty list
        return []

    # If not a list or string, return an empty list
    return []

# Apply the cleaning function to the hashtags column
df['cleaned_hashtags'] = df['hashtags'].apply(clean_hashtags)

# Remove rows where hashtags become empty
df = df[df['cleaned_hashtags'].apply(len) > 0]

# Display the cleaned DataFrame
print(df[['cleaned_hashtags']])

```
Output:

```
                      cleaned_hashtags
0       [python, programming, chatGPT]
1                            [ChatGPT]
4        [Sell, Bybit, Short, SXPUSDT]
7            [ChatGPT, OnlineBusiness]
8                   [politicalscience]
...                                ...
725829                       [ChatGPT]
725831                       [ChatGPT]
725832                       [ChatGPT]
725833                 [ChatGPT, GPT3]
725834                  [ChatGPT, LLM]

[242065 rows x 1 columns]
```

For further analysis (like showing wordcloud) on text feature we need to remove irrelevant symbols, special characters etc.
Refer to the .ipynb for detailed code and output.

## Exploratory Data Analysis (EDA)

### Sentiment Analysis
Sentiment analysis is crucial for understanding the emotional tone behind each tweet. It involves calculating sentiment scores using the VADER Sentiment Analysis tool, which provides a compound score indicating the overall sentiment (positive, negative, or neutral) of each tweet. This analysis helps in categorizing tweets based on their emotional content and understanding trends in public sentiment over time.

Here is number of different sentiment types:

![image](https://github.com/user-attachments/assets/3e40d0d6-8268-4ced-94ec-cc1e5d5ef3fe)

Number of Neutral tweets: 434581
Number of Positive tweets: 162636
Number of Negative tweets: 51831

![image](https://github.com/user-attachments/assets/38b90316-aac9-4db9-81ed-2da4a55e3472)

As we can see maximum tweets are neutral, (more than half). Less tweets are negative.

Percentage % distribution of tweets:

![image](https://github.com/user-attachments/assets/31df111d-f1e4-4b26-a4fa-ff50978b8cdb)



