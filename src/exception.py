import sys
# This function returns the error message with details of the error such as file name and line number where the error occurred
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
    file_name ,exc_tb.tb_lineno ,str(error))
    
    return error_message
# Custom Exception class which will be used to handle exceptions in the project 
class CustomException(Exception):
    def __init__(self, error_message , error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_details)
        
    def __str__(self):
        return self.error_message