import statsmodels.api as sm
from matplotlib import pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
from scipy.stats import t

class regression:
    def __init__(self, df, y:str, X:list):
        """ """
        if isinstance(y, str):
            y = [y]
        # Drop rows with missing values in the columns of interest
        self.df = deepcopy(df.dropna(subset=X+y))
        # Save the labels
        self.labels = (y,X)
        # Turn the data into numpy arrays
        self.y = self.df[y].values
        self.X = self.df[X].values
    def print_min_max(self, column):
        """Print the min and max of a column in the dataframe"""
        print(f"Min: {self.df[column].min()}")
        print(f"Max: {self.df[column].max()}")
    
    def run_regression(self,verbose:bool=True):
        # Run the regression
        est_res_prim = sm.regression.linear_model.OLS(self.y, self.X).fit(cov_type='HC3') 
        
        self.results = {
            'b_hat': est_res_prim.params,
            'se': est_res_prim.bse, 
            'tvalues': est_res_prim.tvalues,
            'pvalues':est_res_prim.pvalues,
            'R2': est_res_prim.rsquared, 
            'R2_adj': est_res_prim.rsquared_adj,
            'nobs': est_res_prim.nobs,
            'fitted': est_res_prim.predict(self.X),
            'residuals': est_res_prim.predict(self.X) - self.y,
            'model': est_res_prim
            }

        if verbose:
            print(print_table(labels=self.labels, results=self.results))
        
        # set up in a dataframe with the labels
        res = pd.DataFrame(index=self.labels[1], columns=['b_hat', 'se', 'tvalues', 'pvalues'], data=self.results)
        res = res.stack().to_frame().rename(columns={0:'value'})
        res.loc[('No. of obs',''),'value'] = self.results['nobs']
        res.loc[('R2',''),'value'] = self.results['R2']
        self.res = res


    def residual_plot(self):
        residuals = self.results['residuals']
        labels = self.labels
        X = self.X

        fig, axs = plt.subplots(len(labels[1])//2,2, figsize=(12,len(labels[1])//2 * 3))

        for i, label in enumerate(labels[1]): 
            if (len(labels[1])%2 != 0):
                if label == 'constant':
                    continue
                else:
                    i = i -1 
            
            # Define the axis
            j = i // 2 
            ax = axs[j,i-j*2]
            
            # Plot residuals against each variable
            ax.scatter(X[:,i], residuals, alpha=0.5);
            
            # Add labels
            ax.set_title(label);
            if i-j*2 == 1:
                ax.spines['left'].set_visible(False)
            else: 
                ax.set_ylabel('Residuals');
            
            # Remove spines
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            # Add grid
            ax.grid(axis='both', linestyle='--', alpha=0.5)

        plt.show()
    

def print_table(
        labels: tuple,
        results: dict,
        title="OLS Regression Results",
        decimals:int=4,
        _lambda:float=None,
        **kwargs
    ) -> None:
    """Prints a nice looking table, must at least have coefficients, 
    standard errors and t-values. The number of coefficients must be the
    same length as the labels.

    Args:
        >> labels (tuple): Touple with first a label for y, and then a list of 
        labels for x.
        >> results (dict): The results from a regression. Needs to be in a 
        dictionary with at least the following keys:
            'b_hat', 'se', 't', 'R2', 'sigma2'
        >> headers (list, optional): Column headers. Defaults to 
        ["", "Beta", "Se", "t-values"].
        >> title (str, optional): Table title. Defaults to "Results".
    """
    
    # Unpack the labels
    label_y, label_x = labels
    assert isinstance(label_x, list), f'label_x must be a list (second part of the tuple, labels)'
    assert len(label_x) == results['b_hat'].size, f'Number of labels for x should be the same as number of estimated parameters'
    
    # Create table, using the label for x to get a variable's coefficient, standard error and t_value.
    cols = ['b_hat', 'se', 'tvalues','pvalues'] # extract these from results
    result_subset = {k:v for k,v in results.items() if k in cols} # subset the results dictionary to just the items we require
    tab = pd.DataFrame(result_subset, index=label_x)
    tab['signi.'] = ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '' for p in tab['pvalues']]
    
    # Print header
    print(f'{title:^80s}')
    print('--'*40)
    print(f"Dependent variable: {label_y}")
    if 'nobs' in results: print(f"Observations: {results['nobs']}")
    print(f"Parameters: {results['b_hat'].size}")
    print('--'*40)
    
    print(tab.round(4))
    # print()
    print('--'*40)
    # Print extra statistics of the model.
    if 'R2' in results: print(f"R2 = {results['R2']:.3f}")
    if 'R2_adj' in results: print(f"R2 Adj = {results['R2_adj']:.3f}")
    if 'sigma2' in results: print(f"sigma2 = {results['sigma2']:.3f}")

class latex_output:
    def __init__(self) :
        self.start_output = r"""\begin{table}[H] \centering """
        self.start_tabular_output = r"""\begin{tabular}{l}"""
        self.model_line = "" # Leave empty
        self.model_dict = {'Model 0': {'Test':(1,2)}}
        self.coefficient_line = ""
        self.t_line = ""
        self.n_line = "No. of obs: " # Initialize row for number of observations
        self.R2_line = "R2: " # Initialize row for R2
        self.end_tabular_output = r"""\end{tabular}"""
        self.end_output = r"""\end{table}"""
    
    def add_model(self, labels, coefficients, se, N:int=None, R2:float=None):
        # Construct the new model's dictionary
        new_model = {label: (coef, err) for label, coef, err in zip(labels, coefficients, se)}

        # Check if the model already exists
        for existing_model in self.model_dict.values():
            if new_model == existing_model:
                print("Duplicate model detected. Skipping addition.")
                return  # Exit without adding the model

        # get the latest model number
        model_number = len(self.model_dict.keys())
        model_name = f"Model {model_number}"

        # add another column 
        self.start_tabular_output = self.start_tabular_output[:-1] + 'c}'
        # add model to coefficients
        self.model_dict[model_name] = new_model
        self.model_line = f"{self.model_line} & {model_name}"

        # add number of observations
        self.n_line = f"{self.n_line} & {N}" if N else self.n_line + ' & '

        # add R2
        self.R2_line = f"{self.R2_line} & {R2:.4f}" if R2 else self.R2_line + ' & '

    def _significance(self, t_value):
        if np.abs(t_value) > t.ppf(0.995, df=1000):
            return '***'
        elif np.abs(t_value) > t.ppf(0.975, df=1000):
            return '** '
        elif np.abs(t_value) > t.ppf(0.95, df=1000):
            return '*  '
        else:
            return '   '

    def create_coefficient_line(self):
        # remove model 0
        if 'Model 0' in self.model_dict:
            del self.model_dict['Model 0']

        # get the latest model number
        max_model_number = len(self.model_dict.keys()) -1 

        unique_labels = set()
        for model in self.model_dict.keys():
            unique_labels.update(self.model_dict[model].keys())
        
        coef_lines, t_lines = [], []
        for label in unique_labels:
            coef_line = f"{label.replace('_',' ')} &"
            t_line = f" &"
            for i, model in enumerate(self.model_dict.keys()):
                if label in self.model_dict[model]:
                    coef, t_value = self.model_dict[model][label]
                    coef_line = coef_line + f" {coef:.4f} &"
                    t_line = t_line + f" ({t_value:.2f}){self._significance(t_value)} &"
                else:
                    coef_line = coef_line + f" &"
                    t_line = t_line + f" &"

                if i == max_model_number: # remove last character
                    coef_line = coef_line[:-1]
                    t_line = t_line[:-1]
            coef_lines.append(coef_line + r"\\")
            t_lines.append(t_line + r"\\")

        return coef_lines, t_lines

    def print_latex(self):
        coef_lines, t_lines = self.create_coefficient_line()

        print(self.start_output)
        print(self.start_tabular_output + ' \\toprule')
        print(self.model_line + ' \\\ \midrule')
        for coef_line, t_line in zip(coef_lines, t_lines):
            print(coef_line)
            print(t_line)
        print('\midrule')
        print(self.n_line + r' \\')
        print(self.R2_line + r' \\')
        print('\\bottomrule')
        print(self.end_tabular_output)

        print(self.end_output)
        