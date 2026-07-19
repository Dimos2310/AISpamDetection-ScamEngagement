INTELLIGENT SYSTEM FOR SPAM EMAIL FILTER 
Methodology:
This research is about to experiment SVM machine learning models and utilize enron spam email data which are publicly available. This research paper aims to achieve the following objectives:
To explore Support Vector Machine algorithms for the spam classifier.
To investigate the workings and phases to develop a machine learning model with the selected datasets
To implement different types of kernel and hyperparameter tuning of SVM model
To test and compare different kernel and hyperparameter which results different accuracy and evaluation metrics outcomes
To implement machine learning framework with Python and utilize Flask to develop a web application to detect spam email.

Scikit-Learn library will be utilized to perform classification tasks onto email spam dataset. One key feature of Scikit-Learn is able to explore a diverse mathematical transformations for capturing intricate patterns within the spam email datasets. Within Scikit-Learn SVM framework, it offers different types of kernel namely linear, polynomial, radial basis function (RBF) and sigmoid.

In order to come up with the machine learning, we have deeply analyzed the function and capability of SVM model. Support Vector Machine (SVM) is a supervised machine learning algorithms that is proven to perform better than some other attendant machine learning algorithms for solving classification problem. The fundamental principle behind SVMs is to find the optimal hyperplane that best separates data points belonging to different classes in a feature space.

However having a SVM classifier model alone to perform spam email detection is not sufficient and not suits for general purpose usage. The below figure (Figure 1.) illustrates our workflow of methodology.



Raw Dataset Description

Dataset is one of the important ingredient of machine learning. Without sufficient data or raw resources, it does not serve any purpose. Dataset is the component that lays the groundwork for machine learning and structure the ability to learn and making accurate predictions. Moreover, it provide a benchmark for measuring the accuracy of machine learning models. With that, we have choosen Enron email dataset which consists 33,716 email messages along with labelled as Spam and Ham. At its raw form, the dataset contain emails that includes several attributes such as sender, recipient, subject line, email body, date and other headers.

Data Preprocessing

After we have input our spam email dataset, the first stage to execute is data preprocessing. This step consists of tokenization, bag-of-words, vectorization and TF-IDF. Tokenization is a process of dividing text into smaller, meaningful units. With utilize NTLK text pre-processing library, it performs normalization, punctation and special character removal. After that, it will reduce words to their base form and handle with infrequent words in a language-defined library. Next, we create a Bag-Of-Words model from vectorization and TF-IDF in order to create numerical representation of text data. Since the content of the email text is divided into tokens, we will generate a vocabulary model and represent it as vectors. With that vectors, we apply TF-IDF to reweigh and


prioritize terms and words that regards as spam. Then the feature selection is taking part to perform selection on the dataset using the feature extraction technique. This technique extracts the relevant features from the email body.

Support Vector Machine Model Training and Validation

Data Splitting Process is a process of splitting the labelled dataset into two parts which are training dataset and the testing dataset. This process begin after the process of data preprocessing, where the dataset is classified after the SVM detected a hyperplane to separate information from classes. The hyperplane itself can be defined as a line or curve that has maximum margin of two classes. After the data splitting process complete, it moves to the model training process. The selected dataset will be trained on the training dataset using the SVM classifier. The dataset learns the relationship between the input and the output during the training session. For data splitting, ratio would be 70:15:15 accordingly to training, testing and validation. The set up training dataset is fed into the SVM model for additional training after the data splitting process. This involves applying the Radial Basis Function (RBF) kernel, which is an essential element that helps the model identify complex patterns in the data. Additionally, hyperparameter tuning is used to improve the performance of the SVM model. This careful tweaking of the hyperparameters guarantees that the model fits the dataset's underlying complexity as best it can. Through the use of this improved model, which has been adjusted in light of the split dataset, our goal is to develop a reliable and precise machine learning model that can successfully classify and predict outcomes on clean, untested data.

Support Vector Machine Model Evaluation

SVM Model Evaluation is the part where the model itself receive the emails. The result of the classification can be seen in the form of binary which is 0 and 1. The binary form indicates the spam and not spam where 0 means not spam and 1 means spam. The performance of the Support Vector Machine was assessed using recall, precision, accuracy, F-measure, and confusion matrix. The purpose of performing evaluation on our machine learning model is to minimize the possibility of misclassification by the evaluation of accuracy score. Misclassification carries significant costs, which makes it extremely crucial matter to be discussed. Erroreously classifying a spam message as ham results in a minor issue, as the only thing that the email user must take action by remove such email. In contrast, when a non-spam message is mistakenly classified as spam, it will be obnoxious as it raises the risk of losing important data due to the filter’s classification error. This is crucial as particularly in configurations where spam messages are automatically removed. Therefore, it is inadequate to evaluate the performance of our model relying solely on accurarcy score.

Design and Development

After successfully constructing our Support Vector Machine (SVM) machine learning and well-evaluated model for spam email detection, the focus has now shifted to the crucial phase of deployment through a user-friendly web application. With leveraging the power of Flask, a lightweight web framework in Python, we are seamlessly integrating our SVM model into an interactive and accessible platform. The web application serves as a function to allow user to place their received email text and predict the indication of a spam email. The deployment process involves encapsulating the trained model using the Pickle framework from Python, which allowing for easy serialization and deserialization, ensuring the model’s persistence and efficiency during runtime. This web application will empower users to efficiently identify and manage spam emails in a real-time environment, enhancing the overall user experience and contributing to a more secure and streamlined communication ecosystem.
Figure 2 shows the interface of the web application


Result/Findings
This research utilized a set of parameters to evaluate the performance of RBF-SVM email spam filtering model. The parameters have been described below:
(a)
Accuracy = ( TP + TN ) / (TP + FN + FP + TN)
Accuracy evaluate based on the number of correct number of emails classified as ‘Spam’ and ‘Ham’ [2]. (b)
Recall = TP / (TP + FN)
Recall measurement provides the calcuation of how many emails were correctly predicted as spam for the total number of spam emails that were provided [2].
(c)
Precision = TP / (TP + FP)
Precision measurement is to calculate the correctly identified values, which is the number of correctly identified spam emails have been classified from the given set of positive emails [2].
(d)
F1 Score = F1 = 2 * (precision * recall) / (precision + recall)
F1-score is the ‘Harmonic Mean’ of the precision and recall values. For all the above evaluation metrics, we utilize a Python Library, Scikit-Learn which provides a comprehensive set of tools for model evaluation and selection, making it convenient to assess the performance of machine learning models [2].
Depending on the kernel function selected, Support Vector Machine (SVM) models can exhibit wide variations in accuracy. In general the underlying features of the dataset and the type of necessary decision boundary have an impact on SVM performance. When the data is well-separated with a linear boundary, linear kernels perform well and achieve high accuracy. Conversely, sigmoid kernels work well for non-linear problems, but the selection of hyperparameters can have an impact on how well they perform. The ability of radial basis function (RBF) kernels to capture intricate relationships in data makes them adaptable and widely used. When handling
non-linear separations, RBF kernels frequently perform better than linear and sigmoid kernels, providing superior accuracy in situations with complex decision boundaries. Below is the summarized table of different kernel applied in our SVM model. The table below (Table 1) describes the different accuracy performance for different kernel and hyperparameter tuning for SVM model. Based of the evaluation, we have conducted deeper analysis with classification report.






