# -*- coding: utf-8 -*-
"""
Task
-------
Nested Logit and MLE

Version      |Author       |Affiliation                |Email
--------------------------------------------------------------------------
Feb 10, 2018 |Chenshuo Sun |Stern Business School, NYU |csun@stern.nyu.edu

Goal(s)
-------
Estimate the model using the Maximum Likelihood
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class Obj(object):
    """Class for Maximum Likelihood Estimation

    Parameters
    ----------
    param1: int
        Use which dataset
    param2: int
        Number of consumers I
    param3: int
        Number of consumers J
    param4: int
        Number of consumers T

    Returns
    -------
    Float
        Log-likelihood
    """

    def __init__(self, m, I, J, T):
        """Class initialization"""
        self.m = m
        self.I = I
        self.J = J
        self.T = T

    def __data_loader(self):
        """Function for loading the data

        Parameters
        ----------
        param1: int
            Dataset No., e.g. m = 1, 2, ..., M

        Returns
        -------
        DataFrame
            The m-th dataset
        """
        m = self.m
        file_name = 'Data_' + str(m) + '.csv'
        print(file_name + ' loaded')
        data = pd.read_csv(file_name)
        return data

    def LL(self, params):
        """Function for computing the log likelihood

        Parameters
        ----------
        param1: int
            Dataset

        Returns
        -------
        Object
            The log likelihood function
        """
        # parameters initialized
        intercept = params[:-2]
        coefficient = params[-2]
        SIG = params[-1]
        #
        I = self.I
        J = self.J
        T = self.T
        # load data
        data = self.__data_loader()
        # compute the Pr(j)
        data['Pr(j)'] = pd.Series(0.0, index=data.index, dtype='float64')
        Z = data.set_index(['I', 'T', 'J'])
        W = Z.groupby(['I', 'T']).size().index.values
        for (i, t) in W:
            temp_sum = 0
            for j in range(1, J + 1):
                temp_sum += np.exp((intercept[j -
                                              1] +
                                    coefficient *
                                    Z.loc[(i, t, j)]['Pjt']) /
                                   SIG)
            for j in range(1, J + 1):
                Z.loc[(i, t, j), 'Pr(j)'] = 1 / \
                (1 + np.power(temp_sum, -SIG)) * \
                    np.exp((intercept[j - 1] + coefficient * \
                            Z.loc[(i, t, j)]['Pjt']) / SIG) / temp_sum
        # compute the log-likelihood, step 1
        LH_it = pd.DataFrame(index=W, columns=['I', 'T', 'lh_it'])
        for (i, t) in W:
            temp_choice = int(
                Z.loc[(i, t, 1), \
                      Z.columns.str.startswith('Yit', na=False)].nonzero()[0])
            if temp_choice == 0:
                temp_Pr0 = 1 - Z.loc[(i, t), 'Pr(j)'].sum()
                lh_it = temp_Pr0
            else:
                temp_Prj = Z.loc[(i, t, temp_choice), 'Pr(j)']
                lh_it = temp_Prj
            LH_it.loc[(i, t), 'lh_it'] = lh_it
            LH_it.loc[(i, t), 'I'] = i
            LH_it.loc[(i, t), 'T'] = t
        # compute the log-likelihood, step 2
        LH_it = LH_it.set_index(['I', 'T'])
        LH_i = pd.DataFrame(index=range(1, I + 1), columns=['I', 'lh_i'])
        for i in range(1, I + 1):
            lh_i = 1
            for t in range(1, T + 1):
                lh_i *= LH_it.loc[(i, t)]
            LH_i.loc[i, 'lh_i'] = lh_i.values[0]
            LH_i.loc[i, 'I'] = i
        # compute the log-likelihood, step 3
        LH_i = LH_i.set_index(['I'])
        LL = 0
        for i in range(1, I + 1):
            LL += np.log(LH_i.loc[i].values[0])
        # return
        # print('...')
        return -LL


def result_show(m, I, J, T, params):
    """Function for Maximum Likelihood Estimation

    Parameters
    ----------
    param1: int
        Use which dataset
    param2: int
        Number of consumers I
    param3: int
        Number of consumers J
    param4: int
        Number of consumers T
    param5: np.array
        The initial value

    Return
    ------
        Print the estimation results
    """
    obj = Obj(m, I, J, T)
    print('It needs some time to estimate...')
    result = minimize(
        obj.LL,
        params,
        method='L-BFGS-B',
        options={
            'gtol': 0.05,
            'disp': True})
    print('Estimation is done on the ' + str(m) + '-th dataset')
    print(result)


def main():
    """Main function
    """
    '''set parameters'''
    m = 3
    I = 500
    J = 2
    T = 50
    # initial values
    params = np.array(np.array([1.0, 1.0, 1.0, 1.0]))
    # print results
    result_show(m, I, J, T, params)


if __name__ == "__main__":
    main()
