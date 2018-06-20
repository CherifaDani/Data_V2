# coding: utf-8
# RECOMMANDATIONS
# Workalendar:
# * pip install __python-dev__
# * pip install __xlrd__
# * pip install __workalendar__
"""
Author: C. DANI
Company: Advestis
Last modification: June, 13 2018
"""

# libraries
import pandas as pd
from datetime import datetime
import logging

# Packages
try:
    import data_utils
except ImportError:
    data_utils = None
    raise ImportError("Don't find the package control_utils")

try:
    import control_utils as cu
except ImportError:
    control_utils = None
    raise ImportError("Don't find the package control_utils")

try:
    from var_logger import setup_logging
except ImportError:
    print ("Don't find the package var_logger")

# Launching the logger module
setup_logging()
logger = logging.getLogger(__name__)
logger.debug('Logger for class ')
# Setting the DEBUG level of the logger
logger.setLevel(logging.INFO)


class Variable(object):
    """
    Class Variable representing a Time Series

    Parameters
    ----------
    script_path : {String type}
                    Derivation script path

    var_type : {String type}
               Variable type, eg: primary, spread ...
               Corresponds to the name of derivation
               script sheets

    var_name : {String type}
               The variable name,
               Corresponds to the sql_name in the
               derivation script

    path : {String type}
            Path to the csv file of the variable

    backup : {Boolean type}
             if True overwrite the csv file of the variable

    rubrique : {String type}
                The category name of the variable

    parents : {list type}
              The parents of the variable, is empty if the
              variable is a primary variable

    path_latest : {string type}
                   Path to retrieve the latest variable,
                   corresponding to a primary variable

    derived_params : {dict type}
                    Parameters of the derivation

    country : {String type}
                The benchmark country of the variable

    val_min : {Float type}
                Minimum value of the variable

    val_max : {Float type}
                Maximum value

    mac_val : {Float type}
                Maximum absolute change

    mccv_val : {Int type}
                Maximum Consecutive Constant Values


     : {Int type}
                Maximum Consecutive missing Values

    last_update : {Date type}
                    The last date when the variable was updated

    Return
    ------
    df : {Pandas dataframe}
            The updated dataframe

    """
    def __init__(self, **parameters):
        for arg, val in parameters.items():
            if str(val).lower() == 'nan':
                setattr(self, arg, None)
            else:
                setattr(self, arg, val)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Class: Variable')

    """------   Getters   -----"""
    def get_param(self, param):
        assert type(param) == str, 'Must be a string'
        return getattr(self, param)

    def get_params(self):
        out = {}
        for key in self.__dict__.keys():
                out[key] = self.__dict__[key]
        return out

    """------   Setters   -----"""
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # path_latest
    def get_latest(self):
        """ Method getting the path of the latest
        variable, and its csv_name from the base
        variable.
        """
        path = self.get_param('path')
        path_latest, csv_name = data_utils.latestpath(path)
        return path_latest, csv_name

    def get_var_state(self):
        var_name = self.get_param('var_name')
        state_path = self.get_param('state_path')
        [var_type, last_update, backup, depth_control, refdate] = data_utils.get_var_state(
                                                state_path, var_name)
        self.set_params(var_type=var_type)
        self.set_params(last_update=last_update)
        self.set_params(backup=backup)
        self.set_params(depth_control=depth_control)
        self.set_params(refdate=refdate)

    """-----  Methods  -----"""
    def __str__(self):
        """
        Method describing the class Variable

        """
        self.get_var_state()
        return "name = {}, path = {}, type = {}".format(
                self.get_param('var_name'),
                self.get_param('path'),
                self.get_param('var_type'))

    def read_var(self, csv_path):
        """
        This function reads a csv file

        Parameters
        ----------
        csv_path : {string type}
                    Path to the csv file

        Return
        ------
        df : {Pandas dataframe}
             The dataframe of the csv file with a sorted time index

        """
        var_name = self.get_param('var_name')
        df = data_utils.load_var(path=csv_path, var_name=var_name)
        return df

    def write_dict(self):
        """
        This function writes some attributes of the variable,
        according to the derivation script!
        """
        var_name = self.get_param('var_name')
        self.get_var_state()
        # Reading the derivation script
        dfs = data_utils.read_deriv_script(self.get_param('script_path'),
                                           self.get_param('var_type'))

        # Retrieving the variable dictionary from dfs
        var_dict = data_utils.fill_dict_from_df(dfs, var_name)

        # Retrieving the parameters of the var from the derivation script
        rubrique = var_dict['cat_name']
        var_name = var_dict['sql_name']
        country = var_dict['country']
        freq = var_dict['freq']
        path = var_dict['path']
        val_min = var_dict['min_val']
        val_max = var_dict['max_val']
        val_mac = var_dict['mac_val']
        mcmv_val = var_dict['mcmv_val']
        mccv_val = var_dict['mccv_val']
        parents = var_dict['parents']
        derived_params = var_dict['parameters']

        # initializing the attributes of the var

        self.set_params(rubrique=rubrique)
        self.set_params(var_name=var_name)
        self.set_params(country=country)
        self.set_params(freq=freq)
        self.set_params(path=path)
        self.set_params(val_mac=val_mac)
        self.set_params(val_max=val_max)
        self.set_params(val_min=val_min)
        self.set_params(mcmv_val=mcmv_val)
        self.set_params(mccv_val=mccv_val)
        self.set_params(parents=parents)
        self.set_params(derived_params=derived_params)

        logger.debug('Dictionary for {} is {} in {}'.format(
                     self.var_name, var_dict, __name__))

        return var_dict

    def control(self):
        """
        Function performing the data control routine on a
        primary variable to update, it does a series of control
        on the DF resulted from the append
        """
        var_dict_df = {'max_ccv': self.get_param('mccv_val'),
                       'freq': self.get_param('freq'),
                       'val_min': self.get_param('val_min'),
                       'val_max': self.get_param('val_max'),
                       'val_mac': self.get_param('val_mac')
                       }

        depth_control = self.get_param('depth_control')
        refdate = self.get_param('refdate')
        path = self.get_param('path')
        path_latest, latest_name = self.get_latest()
        var_name = self.get_param('var_name')

        # Loading the primary variable
        df_base = self.read_var(path)
        df_latest = self.read_var(path_latest)

        # Renaming the column 'Valeur' to the latest name
        df_latest = df_latest.rename(columns={'VALEUR': latest_name})
        base_rows = df_base.shape[0]

        # if refdate, retrieve data from latest file,
        # instead of base file
        if refdate != datetime.today():
            df_base = df_base[df_base.index < refdate]

        # Control the base from a given threshold
        df_base = df_base.iloc[-depth_control:base_rows]
        df_base_tail = df_base.tail(depth_control)
        df = df_base_tail.append(df_latest)

        df, df_info_dict = cu.control_routine(df, df_latest, var_dict_df)
        df = df[~df.index.duplicated(take_last=False)]
        logger.debug('df_info_dict of: {} is: {}'.
                     format(var_name, df_info_dict))
        return df, df_info_dict

    def write_meta_var(self):
        """
        Function writing important informations about
        the variables to a csv file.
        """
        var_name = self.get_param('var_name')
        last_update = self.get_param('last_update')
        backup = self.get_param('backup')
        depth_control = self.get_param('depth_control')
        refdate = self.get_param('refdate')
        var_type = self.get_param('var_type')
        csv_name = self.get_param('state_path')

        cols = ['var_name', 'var_type', 'last_update',
                'backup', 'depth_control', 'refdate']
        backup_int = 0 if backup is True else 1
        var_data = {'var_name': [var_name],
                    'var_type': [var_type],
                    'last_update': [last_update],
                    'backup': [backup_int],
                    'depth_control': [depth_control],
                    'refdate': [refdate]}

        df2 = pd.DataFrame(var_data, columns=cols)
        df2.set_index('var_name', inplace=True)
        df = pd.read_csv(csv_name, index_col='var_name')
        df.loc[var_name] = df2.values[0]
        # Writing dict to a csv file
        df.to_csv(csv_name)

        # Logger
        logger.info('Meta-data Dictionary saved')
        logger.debug('Dictionary saved of: {} is: {}'.
                     format(var_name, var_data))

    def update_prim(self):
        """
        This function updates the primary variable
        """

        backup = self.get_param('backup')
        path = self.get_param('path')
        last_update = self.get_param('last_update')
        var_name = self.get_param('var_name')
        freq = self.get_param('freq')
        refdate = self.get_param('refdate')
        # Verifying if the primary variable is up to day
        update = data_utils.alterfreq(freq, refdate,
                                      last_update, var_name,
                                      path)
        df = pd.DataFrame()
        if update is True:
            logger.debug('Updating primary variable: {}'.format(var_name))

            # Saving the base file
            if backup:
                data_utils.write_zip(path)

            # Loading the primary variable
            df_base = self.read_var(path)
            df_append, df_info_dict = self.control()

            # The dataframe to return, empty if the alert level != 0
            df = pd.DataFrame()

            # alert_level have to be equal to 0 to update var
            if df_info_dict['alert_level'] == 3:
                df = df_base.append(df_append)
                df = df[~df.index.duplicated(take_last=False)]
                # Overwriting primary variable
                data_utils.df_to_csv(df, path)
                # updating the date of the last_update
                last_update = df.index[-1]
                self.set_params(last_update=last_update)
                logger.info('UPDATE DONE, FILE SAVED !!')

                return df, df_info_dict
            else:
                logger.warn('Alert level = {} cannot do the update'.format(
                    df_info_dict['alert_level']))
            logger.debug("Dictionary of current var is: {}".
                         format(self.__dict__))
            return df
        else:
            logger.info('Variable already updated!')
            return df

    def test(self):
        self.get_var_state()
        self.write_dict()
        return self.__dict__

    def update_deriv(self):
        """
        This function reload this class if the variable
        is a derived variable!
        """
        # Variable attributes
        parents = self.get_param('parents')
        var_name = self.get_param('var_name')
        var_type = self.get_param('var_type')
        script_path = self.get_param('script_path')
        backup = self.get_param('backup')
        refdate = self.get_param('refdate')
        depth_control = self.get_param('depth_control')
        last_update = self.get_param('last_update')

        # Retrieving the parents of the variable
        len_parents = len(parents)
        if len_parents != 0:
            for p in range(len_parents):
                logger.info('Direct parents of : {} are: {}'.
                            format(var_name, parents))
                logger.info('Processing parent {}'.format(parents[p]))
                var = Variable(var_name=parents[p], var_type=var_type,
                               script_path=script_path,
                               backup=backup,
                               refdate=refdate,
                               depth_control=depth_control,
                               last_update=last_update
                               )
                # Updating the current variable
                var.update()

    def update(self):
        """
        Function used to update the variable
        """
        var_dict = self.write_dict()
        var_name = self.get_param('var_name')

        logger.debug('The dictionary of the variable {} is: {}'.
                     format(var_name, var_dict))

        # Verifying if the variable is a primary var
        if self.get_param('var_type') == 'Primary':
            df = self.update_prim()
            # Saving the meta-data dictionary
            saved_dict = self.write_meta_var()
            logger.debug('Meta-data Dictionary saved: {}'.format(saved_dict))
            return df
        else:
            self.update_deriv()
