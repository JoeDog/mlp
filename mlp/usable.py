import io
import random
import string

class Util:

  @staticmethod
  def randomNetworkWeight(min=-1, max=1):
    return random.uniform(min,max)  

  @staticmethod
  def randomIndex(size:int)->int:
    return random.randint(0, size-1)

class StringBuffer(object):
  """
  A mutable sequence of characters. A StringBuffer is like a str, but can be modified. 
  At any point in time it contains some particular sequence of characters, but the length 
  and content of the sequence can be changed through certain method calls. 

  """
  def __init__(self) -> None:
    """ 
    Constructor: StringBuffer()

    Parameters:
    None

    Returns:
    StringBuffer
    """
    self._stringio = io.StringIO()

  def append(self, *objects, sep=' ', end='') -> None:
    """
    Append a series of objects to the StringBuffer

    Parameters:
    objects (varied):    A series of objects, primitives, etc. to append to the StringBuffer
    sep(char):           [Optional] A character to separate the objects Default: ' '
    end(char):           [Optional] A character to end the objects. Default '' Generally a newline ('\n')

    Returns:
    StringBuffer
    """
    print(*objects, sep=sep, end=end, file=self._stringio)

  def substring(self, begin: int, end: int) -> str:
    """
    Returns a new String that contains a subsequence of characters currently contained within 'begin' and 'end'.

    Parameters:
    begin(int):          The starting position of the substring
    end  (int):          The ending position of the substring

    Returns:
    substring(str):      A substring between (and including) the charaters from 'begin' to 'end'
    """
    if begin > end:
      raise ValueError("\nUsage: sb.substring(begin, end)\nError: begin cannot be greater than end")
    string = self._stringio.getvalue()
    first  = begin if begin > 0 and begin < len(string) else 0
    last   = end   if end   < len(string) else len(string)
    return string[first:last]

  def toString(self) -> str:
    """
    Returns a string representing the data in this sequence.

    Parameters:
    None

    Returns:
    String               A string representation of this series of characters.
    """
    return self.__str__()
 
  def __str__(self) -> str:
    return self._stringio.getvalue()


