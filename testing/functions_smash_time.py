import numpy as np
import math
import datetime



def duration_to_timedelta(duration:list|int|float):
    
    if isinstance(duration,int|float):
        delta_t=datetime.timedelta(seconds=duration)
    elif isinstance(duration,list):
        
        if (isinstance(duration[0],str)):
            unit=duration[0]
            time=duration[1]
        elif (isinstance(duration[1],str)):
            unit=duration[1]
            time=duration[0]
        else:
            raise ValueError(
                f"duration '{duration}' must contain a str and and integer"
            )
        
        if isinstance(time,int|float):
        
            if (unit=="seconds") | (unit=="s"):
                delta_t=datetime.timedelta(seconds=time)
            elif (unit=="minutes") | (unit=="min"):
                delta_t=datetime.timedelta(minutes=time)
            elif (unit=="hours") | (unit=="h"):
                delta_t=datetime.timedelta(hours=time)
            elif (unit=="days") | (unit=="d"):
                delta_t=datetime.timedelta(days=time)
            else:
                raise ValueError(
                    f"duration unit '{unit}' must be an str. Possible values: (seconds|s) (minutes|min) (hours|h) (days|d)"
                )
        else:
            raise ValueError(
                f"duration value '{time}' must be an integer or float."
            )
    else:
        raise ValueError(
            f"duration '{duration}' must be a list or an integer or float."
            )
    
    return delta_t


def to_datetime(time=''):
    if isinstance(time,str):
        return datetime.datetime.fromisoformat(time)
    else:
        raise ValueError(
            f"time '{time}' must be a instance of str."
            )


def to_datestring(date):
    return date.strftime("%Y-%m-%d %H:%M")


def dict_filter_by_date(in_dict,t_start=None,t_end=None):
    
    out_res=dict()
    
    if t_start is not None:
        t_s=datetime.datetime.fromisoformat(t_start)
    
    if t_end is not None:
        t_e=datetime.datetime.fromisoformat(t_end)
    
    for key,value in in_dict.items():
        
        date_simu=datetime.datetime.fromisoformat(key)
        
        if t_start is None:
            t_s=date_simu
        
        if t_end is None:
            t_e=date_simu
        
        if (date_simu>=t_s) and (date_simu<=t_e):
            out_res.update({key:value})
            
    return out_res


def stringdecode(self):
    """
    Decode characters from a array of integer: Usefull when you try to access to a array of string in the object model.
    """
    return self.tobytes(order='F').decode('utf-8').split()



#date and time functions
def date_to_path(date, format_schapi=True):
    """
    Convert the SMASH date format to a path for searching rainfall
    
    Parameters
    ----------
    date : integer representing a date with the format %Y%m%d%H%M%S
    
    Returns
    ----------
    path : string representing the path /year/month/day/
    
    Examples
    ----------
    date_to_path(date.strftime('%Y%m%d%H%M')
    /%Y/%m/%d/
    """
    year=date[0:4]
    month=date[4:6]
    day=date[6:8]
    
    if format_schapi:
        
        ret = os.sep + year + os.sep + month + os.sep + day + os.sep
    
    else:
        
        ret = os.sep + year + os.sep + month + os.sep
    
    return ret


def decompose_date(date):
    """
    Split a SMASH date
    
    Parameters
    ----------
    date : integer representing a date with the format %Y%m%d%H%M%S
    
    Returns
    ----------
    year,month,day,hour,minute : integers each part of the date (seconds not included)
    
    Examples
    ----------
    year,month,day,hour,minute=decompose_date(date.strftime('%Y%m%d%H%M')
    """
    year=date[0:4]
    month=date[4:6]
    day=date[6:8]
    hour=date[8:10]
    minute=date[10:13]
    return year,month,day,hour,minute


def date_range(self):
    """
    Generate a  Panda date list according the smash model setup
    
    Parameters
    ----------
    self : object model
    
    Returns
    ----------
    date_list: a Panda list of date from self.setup.date_deb to self.setup.date_prv
    
    Examples
    ----------
    model = smash.Model(configuration='Data/Real_case/configuration.txt')
    date_list=date_range(model)
    """
    delta_t=datetime.timedelta(seconds=self.setup.dt)
    
    year,month,day,hour,minute=decompose_date(self.setup.date_deb.decode())
    date_start = datetime.datetime(int(year),int(month),int(day),int(hour),int(minute))+delta_t
    
    year,month,day,hour,minute=decompose_date(self.setup.date_prv.decode())
    date_end = datetime.datetime(int(year),int(month),int(day),int(hour),int(minute))
    
    date_list=pandas.date_range(date_start,date_end,freq=delta_t)
    return date_list
    
