import matplotlib.pyplot as plt
from wordcloud import WordCloud
from functools import reduce
import numpy as np


def plot_word_clouds(classifiers, feature_names_list, top_features=20):
    f, axs = plt.subplots(len(classifiers), 2, figsize=(50, 25))
    for classifier, feature_names, ax in zip(classifiers, feature_names_list, axs):
        coef = classifier.coef_.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        positives = [int(1000 * x) for x in coef[top_positive_coefficients]]
        negatives = [abs(int(1000 * x)) for x in coef[top_negative_coefficients]]
        positive_text = ' '.join(
            reduce(
                lambda x, y: x + y,
                [[pos_word] * pos_coef for pos_word, pos_coef in zip(np.array(feature_names)[top_positive_coefficients], positives)]
            )
        )
        negative_text = ' '.join(
            reduce(
                lambda x, y: x + y,
                [[neg_word] * neg_coef for neg_word, neg_coef in zip(np.array(feature_names)[top_negative_coefficients], negatives)]
            )
        )
        pos_wordcloud = WordCloud(collocations=False, max_words=top_features).generate(positive_text)
        neg_wordcloud = WordCloud(collocations=False, max_words=top_features).generate(negative_text)
        ax[0].imshow(pos_wordcloud, interpolation="bilinear")
        ax[0].axis("off")
        ax[0].set_title('Top-{} positive words'.format(top_features), fontsize=45)
        ax[1].imshow(neg_wordcloud, interpolation="bilinear")
        ax[1].axis("off")
        ax[1].set_title('Top-{} negative words'.format(top_features), fontsize=45)
    plt.tight_layout()
    plt.show()


def plot_coefficients(classifiers, feature_names_list, top_features=20):
    f, axs = plt.subplots(1, len(classifiers), sharey=False, sharex=False, figsize=(30, 50))
    for classifier, feature_names, ax in zip(classifiers, feature_names_list, axs):
        coef = classifier.coef_.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        ax.barh(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        ax.set_yticks(np.arange(2 * top_features))
        ax.set_yticklabels(feature_names[top_coefficients], fontsize=20, color='green')
    plt.tight_layout()
    plt.show()
