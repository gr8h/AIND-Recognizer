import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str, n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)

class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """

        """
        * where L is the likelihood of the fitted model, p is the number of parameters,and N is the number of data points
        
        Initial state occupation probabilities = num_states
        Transition probabilities = num_states*(num_states - 1)
        Emission probabilities = num_states*numFeatures*2 = numMeans+numCovars
        
        numMeans and numCovars are the number of means and covars calculated. 
        One mean and covar for each state and features. 
        Then the total number of parameters are:
            Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities
        """

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_selected_score = float('-inf')
        best_selected_model = None
        n = len(self.X)
        num_features = self.X.shape[1]

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:

                model = self.base_model(num_states)
                logL = model.score(self.X, self.lengths)
                logN = np.log(n)

                transition_probability = num_states * (num_states - 1)
                emission_probability = num_states * num_features * 2

                p = num_states + transition_probability + emission_probability

                curr_score = -2 * logL + p * logN

                if curr_score > best_selected_score:
                    best_selected_score = curr_score
                    best_selected_model = model

            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
                pass

        return best_selected_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_selected_score = float('-inf')
        best_selected_model = None
        n = len(self.X)

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_states)
                logL = model.score(self.X, self.lengths)

                word_scores = []
                logL_all_but_word = 0

                for word in [w for w in self.words if w != self.this_word]:
                    word_X, word_lengths = self.hwords[word]
                    word_score = model.score(word_X, word_lengths)
                    word_scores.append(word_score)

                if word_scores:
                    logL_all_but_word = sum(word_scores)/len(word_scores)

                curr_score = logL - logL_all_but_word
                if curr_score > best_selected_score:
                    best_selected_score = curr_score
                    best_selected_model = model


            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
                pass

        return best_selected_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if len(self.sequences) < 2:
            return None

        if len(self.sequences) == 2:
            n_splits = 2
        else:
            n_splits = 3

        best_selected_score = float('-inf')
        best_selected_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):

            try:
                curr_scores = []

                kf = KFold(n_splits=n_splits)
                for cv_train_index, cv_test_index in kf.split(self.sequences):

                    train_X, train_lengths = combine_sequences(cv_train_index, self.sequences)
                    test_X, test_lengths = combine_sequences(cv_test_index, self.sequences)

                    model = GaussianHMM(n_components=num_states
                                        , covariance_type="diag"
                                        , n_iter=1000
                                        , random_state=self.random_state
                                        , verbose=False).fit(train_X, train_lengths)

                    curr_score = model.score(test_X, test_lengths)
                    curr_scores.append(curr_score)

                avg_score = sum(curr_scores)/len(curr_scores)

                if avg_score > best_selected_score:
                        best_selected_score = avg_score
                        best_selected_model = model

            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
                pass

        return best_selected_model
