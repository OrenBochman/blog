Computational Linguistics tasks:

create a corpus

clean it up

create a vocabulary

create a frequency list

create a term document matrix TDF

list n-grams

generate word clouds

mine TDF it for collocations

similarity

- cosine similarity
- TDIDF
- Nearest neighbor word clustering

embedding

word embedding

sentence embedding

concordance

- KWIC, keywords in context
- KWOC, keywords out of context

# Setup

``` r
require_install <- function(libs) {

    for (i in libs){
        if( !is.element(i, .packages(all.available = TRUE)) ) {
            install.packages(i)
        }
        library(i,character.only = TRUE)
        }
}

require_install(libs=c('tm','SnowballC','tidytext','dplyr','wordcloud'))
```

    # Downloading packages -------------------------------------------------------
    - Downloading tm from CRAN ...                  OK [313.3 Kb in 0.52s]
    - Downloading slam from CRAN ...                OK [53.7 Kb in 0.18s]
    - Downloading NLP from CRAN ...                 OK [145.3 Kb in 0.21s]
    - Downloading BH from CRAN ...                  OK [13.7 Mb in 2.6s]
    Successfully downloaded 4 packages in 4.8 seconds.

    The following package(s) will be installed:
    - BH   [1.87.0-1]
    - NLP  [0.3-2]
    - slam [0.1-55]
    - tm   [0.7-16]
    These packages will be installed into "~/work/blog/renv/library/R-4.5/x86_64-pc-linux-gnu".

    # Installing packages --------------------------------------------------------
    - Installing slam ...                           OK [built from source and cached in 2.3s]
    - Installing NLP ...                            OK [built from source and cached in 1.8s]
    - Installing BH ...                             OK [built from source and cached in 3.3s]
    - Installing tm ...                             OK [built from source and cached in 7.2s]
    Successfully installed 4 packages in 15 seconds.

    Loading required package: NLP

    Attaching package: 'dplyr'

    The following objects are masked from 'package:stats':

        filter, lag

    The following objects are masked from 'package:base':

        intersect, setdiff, setequal, union

    Loading required package: RColorBrewer

# Corpus

``` r
doc1 <- "drugs, hospitals, doctors"
doc2 <- "smog, pollution, micro-plastics, environment."
doc3 <- "doctors, hospitals, healthcare"
doc4 <- "pollution, environment, water."
doc5 <- "I love NLP with deep learning."
doc6 <- "I love machine learning."
doc7 <- "He said he was keeping the wolf from the door."
doc8 <- "Time flies like an arrow, fruit flies like a banana."
doc9 <- "pollution, greenhouse gasses, GHG, hydrofluorocarbons, ozone hole, global warming. Montreal Protocol."
doc10 <- "greenhouse gasses, hydrofluorocarbons, perfluorocarbons, sulfur hexafluoride, carbon dioxide, carbon monoxide, CO2, hydrofluorocarbons, methane, nitrous oxide."
corpus <- c(doc1, doc2, doc3, doc4,doc5, doc6,doc7,doc8,doc9,doc10)   # <1>
tm_corpus <- Corpus(VectorSource(corpus))                       # <2>
```

1.  concat docs into corpus var
2.  created a corpus of class Corpus from the corpus var

Next, let’s inspect the corpus

``` r
inspect(tm_corpus) # <3>
```

1.  inspect the corpus

    <<SimpleCorpus>>
    Metadata:  corpus specific: 1, document level (indexed): 0
    Content:  documents: 10

     [1] drugs, hospitals, doctors                                                                                                                                      
     [2] smog, pollution, micro-plastics, environment.                                                                                                                  
     [3] doctors, hospitals, healthcare                                                                                                                                 
     [4] pollution, environment, water.                                                                                                                                 
     [5] I love NLP with deep learning.                                                                                                                                 
     [6] I love machine learning.                                                                                                                                       
     [7] He said he was keeping the wolf from the door.                                                                                                                 
     [8] Time flies like an arrow, fruit flies like a banana.                                                                                                           
     [9] pollution, greenhouse gasses, GHG, hydrofluorocarbons, ozone hole, global warming. Montreal Protocol.                                                          
    [10] greenhouse gasses, hydrofluorocarbons, perfluorocarbons, sulfur hexafluoride, carbon dioxide, carbon monoxide, CO2, hydrofluorocarbons, methane, nitrous oxide.

## Text preprocessing

``` r
tm_corpus <- tm_map(tm_corpus, tolower) # <4>
inspect(tm_corpus)
```

1.  this makes all the tokens lowercase

    <<SimpleCorpus>>
    Metadata:  corpus specific: 1, document level (indexed): 0
    Content:  documents: 10

     [1] drugs, hospitals, doctors                                                                                                                                      
     [2] smog, pollution, micro-plastics, environment.                                                                                                                  
     [3] doctors, hospitals, healthcare                                                                                                                                 
     [4] pollution, environment, water.                                                                                                                                 
     [5] i love nlp with deep learning.                                                                                                                                 
     [6] i love machine learning.                                                                                                                                       
     [7] he said he was keeping the wolf from the door.                                                                                                                 
     [8] time flies like an arrow, fruit flies like a banana.                                                                                                           
     [9] pollution, greenhouse gasses, ghg, hydrofluorocarbons, ozone hole, global warming. montreal protocol.                                                          
    [10] greenhouse gasses, hydrofluorocarbons, perfluorocarbons, sulfur hexafluoride, carbon dioxide, carbon monoxide, co2, hydrofluorocarbons, methane, nitrous oxide.

``` r
tm_corpus <- tm_map(tm_corpus, content_transformer(removePunctuation)) # <5>
inspect(tm_corpus)
```

1.  this removes punctuation tokens

    <<SimpleCorpus>>
    Metadata:  corpus specific: 1, document level (indexed): 0
    Content:  documents: 10

     [1] drugs hospitals doctors                                                                                                                              
     [2] smog pollution microplastics environment                                                                                                             
     [3] doctors hospitals healthcare                                                                                                                         
     [4] pollution environment water                                                                                                                          
     [5] i love nlp with deep learning                                                                                                                        
     [6] i love machine learning                                                                                                                              
     [7] he said he was keeping the wolf from the door                                                                                                        
     [8] time flies like an arrow fruit flies like a banana                                                                                                   
     [9] pollution greenhouse gasses ghg hydrofluorocarbons ozone hole global warming montreal protocol                                                       
    [10] greenhouse gasses hydrofluorocarbons perfluorocarbons sulfur hexafluoride carbon dioxide carbon monoxide co2 hydrofluorocarbons methane nitrous oxide

``` r
tm_corpus <- tm_map(tm_corpus, removeWords, stopwords("english")) # <6>
inspect(tm_corpus)
```

1.  this removes stop words

    <<SimpleCorpus>>
    Metadata:  corpus specific: 1, document level (indexed): 0
    Content:  documents: 10

     [1] drugs hospitals doctors                                                                                                                              
     [2] smog pollution microplastics environment                                                                                                             
     [3] doctors hospitals healthcare                                                                                                                         
     [4] pollution environment water                                                                                                                          
     [5]  love nlp  deep learning                                                                                                                             
     [6]  love machine learning                                                                                                                               
     [7]  said   keeping  wolf   door                                                                                                                         
     [8] time flies like  arrow fruit flies like  banana                                                                                                      
     [9] pollution greenhouse gasses ghg hydrofluorocarbons ozone hole global warming montreal protocol                                                       
    [10] greenhouse gasses hydrofluorocarbons perfluorocarbons sulfur hexafluoride carbon dioxide carbon monoxide co2 hydrofluorocarbons methane nitrous oxide

``` r
tm_corpus <- tm_map(tm_corpus, removeNumbers) # <7>
inspect(tm_corpus)
```

1.  this removes numbers

    <<SimpleCorpus>>
    Metadata:  corpus specific: 1, document level (indexed): 0
    Content:  documents: 10

     [1] drugs hospitals doctors                                                                                                                             
     [2] smog pollution microplastics environment                                                                                                            
     [3] doctors hospitals healthcare                                                                                                                        
     [4] pollution environment water                                                                                                                         
     [5]  love nlp  deep learning                                                                                                                            
     [6]  love machine learning                                                                                                                              
     [7]  said   keeping  wolf   door                                                                                                                        
     [8] time flies like  arrow fruit flies like  banana                                                                                                     
     [9] pollution greenhouse gasses ghg hydrofluorocarbons ozone hole global warming montreal protocol                                                      
    [10] greenhouse gasses hydrofluorocarbons perfluorocarbons sulfur hexafluoride carbon dioxide carbon monoxide co hydrofluorocarbons methane nitrous oxide

``` r
tm_corpus <- tm_map(tm_corpus, stemDocument, language="english") # <8>
inspect(tm_corpus)
```

1.  this stems the words

    <<SimpleCorpus>>
    Metadata:  corpus specific: 1, document level (indexed): 0
    Content:  documents: 10

     [1] drug hospit doctor                                                                                                                       
     [2] smog pollut microplast environ                                                                                                           
     [3] doctor hospit healthcar                                                                                                                  
     [4] pollut environ water                                                                                                                     
     [5] love nlp deep learn                                                                                                                      
     [6] love machin learn                                                                                                                        
     [7] said keep wolf door                                                                                                                      
     [8] time fli like arrow fruit fli like banana                                                                                                
     [9] pollut greenhous gass ghg hydrofluorocarbon ozon hole global warm montreal protocol                                                      
    [10] greenhous gass hydrofluorocarbon perfluorocarbon sulfur hexafluorid carbon dioxid carbon monoxid co hydrofluorocarbon methan nitrous oxid

``` r
tm_corpus <- tm_map(tm_corpus, stripWhitespace) # <9>
inspect(tm_corpus)
```

1.  Removing Whitespaces - a single white space or group of whitespaces may be considered to be a token within a corpus. This is how we remove these token

    <<SimpleCorpus>>
    Metadata:  corpus specific: 1, document level (indexed): 0
    Content:  documents: 10

     [1] drug hospit doctor                                                                                                                       
     [2] smog pollut microplast environ                                                                                                           
     [3] doctor hospit healthcar                                                                                                                  
     [4] pollut environ water                                                                                                                     
     [5] love nlp deep learn                                                                                                                      
     [6] love machin learn                                                                                                                        
     [7] said keep wolf door                                                                                                                      
     [8] time fli like arrow fruit fli like banana                                                                                                
     [9] pollut greenhous gass ghg hydrofluorocarbon ozon hole global warm montreal protocol                                                      
    [10] greenhous gass hydrofluorocarbon perfluorocarbon sulfur hexafluorid carbon dioxid carbon monoxid co hydrofluorocarbon methan nitrous oxid

``` r
dtm <- DocumentTermMatrix(tm_corpus)
inspect(dtm)
```

    <<DocumentTermMatrix (documents: 10, terms: 43)>>
    Non-/sparse entries: 53/377
    Sparsity           : 88%
    Maximal term length: 17
    Weighting          : term frequency (tf)
    Sample             :
        Terms
    Docs doctor environ fli gass hospit hydrofluorocarbon learn like love pollut
      1       1       0   0    0      1                 0     0    0    0      0
      10      0       0   0    1      0                 2     0    0    0      0
      2       0       1   0    0      0                 0     0    0    0      1
      3       1       0   0    0      1                 0     0    0    0      0
      4       0       1   0    0      0                 0     0    0    0      1
      5       0       0   0    0      0                 0     1    0    1      0
      6       0       0   0    0      0                 0     1    0    1      0
      7       0       0   0    0      0                 0     0    0    0      0
      8       0       0   2    0      0                 0     0    2    0      0
      9       0       0   0    1      0                 1     0    0    0      1

``` r
findFreqTerms(dtm, 2)
```

     [1] "doctor"            "hospit"            "environ"          
     [4] "pollut"            "learn"             "love"             
     [7] "fli"               "like"              "gass"             
    [10] "greenhous"         "hydrofluorocarbon" "carbon"           

``` r
findAssocs(dtm, "polution", 0.8)
```

    $polution
    numeric(0)

``` r
as.matrix(dtm)
```

        Terms
    Docs doctor drug hospit environ microplast pollut smog healthcar water deep
      1       1    1      1       0          0      0    0         0     0    0
      2       0    0      0       1          1      1    1         0     0    0
      3       1    0      1       0          0      0    0         1     0    0
      4       0    0      0       1          0      1    0         0     1    0
      5       0    0      0       0          0      0    0         0     0    1
      6       0    0      0       0          0      0    0         0     0    0
      7       0    0      0       0          0      0    0         0     0    0
      8       0    0      0       0          0      0    0         0     0    0
      9       0    0      0       0          0      1    0         0     0    0
      10      0    0      0       0          0      0    0         0     0    0
        Terms
    Docs learn love nlp machin door keep said wolf arrow banana fli fruit like time
      1      0    0   0      0    0    0    0    0     0      0   0     0    0    0
      2      0    0   0      0    0    0    0    0     0      0   0     0    0    0
      3      0    0   0      0    0    0    0    0     0      0   0     0    0    0
      4      0    0   0      0    0    0    0    0     0      0   0     0    0    0
      5      1    1   1      0    0    0    0    0     0      0   0     0    0    0
      6      1    1   0      1    0    0    0    0     0      0   0     0    0    0
      7      0    0   0      0    1    1    1    1     0      0   0     0    0    0
      8      0    0   0      0    0    0    0    0     1      1   2     1    2    1
      9      0    0   0      0    0    0    0    0     0      0   0     0    0    0
      10     0    0   0      0    0    0    0    0     0      0   0     0    0    0
        Terms
    Docs gass ghg global greenhous hole hydrofluorocarbon montreal ozon protocol
      1     0   0      0         0    0                 0        0    0        0
      2     0   0      0         0    0                 0        0    0        0
      3     0   0      0         0    0                 0        0    0        0
      4     0   0      0         0    0                 0        0    0        0
      5     0   0      0         0    0                 0        0    0        0
      6     0   0      0         0    0                 0        0    0        0
      7     0   0      0         0    0                 0        0    0        0
      8     0   0      0         0    0                 0        0    0        0
      9     1   1      1         1    1                 1        1    1        1
      10    1   0      0         1    0                 2        0    0        0
        Terms
    Docs warm carbon dioxid hexafluorid methan monoxid nitrous oxid perfluorocarbon
      1     0      0      0           0      0       0       0    0               0
      2     0      0      0           0      0       0       0    0               0
      3     0      0      0           0      0       0       0    0               0
      4     0      0      0           0      0       0       0    0               0
      5     0      0      0           0      0       0       0    0               0
      6     0      0      0           0      0       0       0    0               0
      7     0      0      0           0      0       0       0    0               0
      8     0      0      0           0      0       0       0    0               0
      9     1      0      0           0      0       0       0    0               0
      10    0      2      1           1      1       1       1    1               1
        Terms
    Docs sulfur
      1       0
      2       0
      3       0
      4       0
      5       0
      6       0
      7       0
      8       0
      9       0
      10      1

load(url(“https://cbail.github.io/Trump_Tweets.Rdata”)) head(trumptweets\$text)

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2011,
  author = {Bochman, Oren},
  title = {Text {Mining} {With} {R}},
  date = {2011-11-29},
  url = {https://orenbochman.github.io/posts/2011/2011-11-29-text-mining-with-r/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2011. “Text Mining With R.” November 29. <https://orenbochman.github.io/posts/2011/2011-11-29-text-mining-with-r/>.
