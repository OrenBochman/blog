## Text2topic Leverage reviews data for multi-label topics classification in Booking.com - Moran Beladev & Elina Frayerman

### Abstract:

Having millions of customer reviews, we would like to better understand them and leverage this data for different use cases. For example, finding popular activities per destination, detecting popular facilities per property, allowing the users to filter reviews by specific topics, detecting violence in reviews and summarizing most discussed topics per property.

In this talk, we will present how we build a multilingual multi-label topic classification model that supports zero-shot, to match reviews with unseen users’ search topics.

We will show how fine-tuning BERT-like models on the tourism domain with a small dataset can outperform other pre-trained models and will share experiment results of different architectures.

Furthermore, we will present how we collected the data using an active learning approach and AWS Sagemaker ground truth tool, and we will show a short demo of the model with explainability using Streamlit.

### Moran Beladev Bio:

Moran is a machine learning manager at booking.com, researching and developing computer vision and NLP models for the tourism domain. Moran is a Ph.D candidate in information systems engineering at Ben Gurion University, researching NLP aspects in temporal graphs. Previously worked as a Data Science Team Leader at Diagnostic Robotics, building ML solutions for the medical domain and NLP algorithms to extract clinical entities from medical visit summaries.

# An error occurred.

Unable to execute JavaScript.

### Slides

[![slide](session1/ss001.png)](session1/ss001.png "slide")

slide

[![slide](session1/ss002.png)](session1/ss002.png "slide")

slide

[![slide](session1/ss003.png)](session1/ss003.png "slide")

slide

[![slide](session1/ss004.png)](session1/ss004.png "slide")

slide

[![What is CIP](session1/ss005.png)](session1/ss005.png "What is CIP")

What is CIP

[![What is CIP](session1/ss006.png)](session1/ss006.png "What is CIP")

What is CIP

[![Text2Topic](session1/ss007.png)](session1/ss007.png "Text2Topic")

Text2Topic

[![Overview](session1/ss008.png)](session1/ss008.png "Overview")

Overview

[![Data Sources](session1/ss009.png)](session1/ss009.png "Data Sources")

Data Sources

[![Data Sources](session1/ss010.png)](session1/ss010.png "Data Sources")

Data Sources

[![Data Sources](session1/ss011.png)](session1/ss011.png "Data Sources")

Data Sources

[![Data Sources](session1/ss012.png)](session1/ss012.png "Data Sources")

Data Sources

[![Data Sources](session1/ss013.png)](session1/ss013.png "Data Sources")

Data Sources

[![Motivation/Goals](session1/ss014.png)](session1/ss014.png "Motivation/Goals")

Motivation/Goals

[![slide](session1/ss015.png)](session1/ss015.png "slide")

slide

[![How it Works?](session1/ss016.png)](session1/ss016.png "How it Works?")

How it Works?

[![Cross Encoder architecture](session1/ss017.png)](session1/ss017.png "Cross Encoder architecture")

Cross Encoder architecture

[![Cross Encoder architecture](session1/ss018.png)](session1/ss018.png "Cross Encoder architecture")

Cross Encoder architecture

[![Bi-Encoder architecture](session1/ss019.png)](session1/ss019.png "Bi-Encoder architecture")

Bi-Encoder architecture

[![Bi-Encoder architecture](session1/ss020.png)](session1/ss020.png "Bi-Encoder architecture")

Bi-Encoder architecture

[![Bi-Encoder architecture](session1/ss021.png)](session1/ss021.png "Bi-Encoder architecture")

Bi-Encoder architecture

[![Bi-Encoder architecture](session1/ss022.png)](session1/ss022.png "Bi-Encoder architecture")

Bi-Encoder architecture

[![Bi-Encoder self-supervised](session1/ss023.png)](session1/ss023.png "Bi-Encoder self-supervised")

Bi-Encoder self-supervised

[![Main Differences](session1/ss024.png)](session1/ss024.png "Main Differences")

Main Differences

[![Dynamic Padding](session1/ss025.png)](session1/ss025.png "Dynamic Padding")

Dynamic Padding

[![Dynamic Padding](session1/ss026.png)](session1/ss026.png "Dynamic Padding")

Dynamic Padding

[![Dynamic Padding](session1/ss027.png)](session1/ss027.png "Dynamic Padding")

Dynamic Padding

[![Evaluation](session1/ss028.png)](session1/ss028.png "Evaluation")

Evaluation

[![Results](session1/ss029.png)](session1/ss029.png "Results")

Results

[![Metrics](session1/ss030.png)](session1/ss030.png "Metrics")

Metrics

[![Results](session1/ss031.png)](session1/ss031.png "Results")

Results

- note Muse-large used as a baseline!

[![slide](session1/ss032.png)](session1/ss032.png "slide")

slide

[![slide](session1/ss033.png)](session1/ss033.png "slide")

slide

[![slide](session1/ss034.png)](session1/ss034.png "slide")

slide

Well done! They did the experiment way past the point where the effects maxed. The main takeaway here is that 100 docs suffice for getting good results on a new topic.

[![slide](session1/ss035.png)](session1/ss035.png "slide")

slide

[![slide](session1/ss036.png)](session1/ss036.png "slide")

slide

[![slide](session1/ss037.png)](session1/ss037.png "slide")

slide

Great talk - the padding tip is probably worth the price of admission :-)

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2023,
  author = {Bochman, Oren},
  title = {Text2topic - {Leverage} Reviews Data for Multi-Label Topics
    Classification in {Booking.com}},
  date = {2023-02-28},
  url = {https://orenbochman.github.io/posts/2023/02-28-nlp-il-booking-meetup/NLP-IL-Booking Text2Topic.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2023. “Text2topic - Leverage Reviews Data for Multi-Label Topics Classification in Booking.com.” February 28. [https://orenbochman.github.io/posts/2023/02-28-nlp-il-booking-meetup/NLP-IL-Booking Text2Topic.html](https://orenbochman.github.io/posts/2023/02-28-nlp-il-booking-meetup/NLP-IL-Booking%20Text2Topic.html).
