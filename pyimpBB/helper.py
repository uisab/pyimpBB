'''This Modul contains helper classes and functions for use in all other models of this package and beyond.'''
import numpy as np
from interval import interval, imath

#helper classes
#-------------------------------------------------------------------------------------------------------------
def vecoperator(func):
    def wrapper(self,other):
        if isinstance(other,obvec):
            if len(self) == len(other):
                return type(self)((func(s,o) for s,o in zip(self,other)))
            elif len(other) == 1:   
                return type(self)((func(s,other[0]) for s in self))
            elif len(self) == 1:   
                return type(self)((func(self[0],o) for o in other))
            else:
                raise ValueError("operands could not be broadcast together with shapes " +str(len(self))+" != "+str(len(other)))
        else:
            return type(self)((func(s,other) for s in self))
    return wrapper

class obvec(tuple):
    """A vector consisting of arbitrary objects.

    An object-vector is an immutable object that is created by specifying 
    its components:

        >>> obvec([O1,O2,O3])
        obvec([O1,O2,O3])

    constructs an object-vector whose entries are references to the objects Oi, where i corresponds to the row. 
    Casting into and back from ndarray from the numpy package is supported:

        >>> numpy.array(obvec(numpy.array([1,4,3])))
        array([1, 4, 3])
    
    All base operations on object-vectors are assigned to the entries in a vector-like manner. 
    In addition, some vector operations known from the numpy package, such as dot product or transpose, are implemented.

        >>> (1 + obvec([2,4])) / obvec([-1,2]) @ obvec([3,1])
        -6.5

    """

    #Unary operators and functions
    def __pos__(self):
        return self
    
    def __neg__(self):
        return type(self)((-s for s in self))
    
    def __abs__(self):
        return type(self)((s.abs() for s in self))
    
    def __invert__(self):
        return type(self)((s.invert() for s in self))
    
    def __round__(self,n):
        return type(self)((s.round(n) for s in self))
    
    #Normal/Reflected arithmetic operators
    @vecoperator
    def __add__(self, other):
        return self+other
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self,other):
        return self.__add__(other.__neg__())
    
    def __rsub__(self,other):
        return (self.__neg__()).__add__(other)
    
    @vecoperator
    def __mul__(self,other):
        return self*other
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    @vecoperator
    def __div__(self,other):
        return self/other
    
    __truediv__ = __div__
    
    def __rdiv__(self,other):
        return self.__div__(other)
    
    __rtruediv__ = __rdiv__

    @vecoperator
    def __mod__(self,other):
        return self%other
    
    def __rmod__(self,other):
        return self.__mod__(other)
        
    def __pow__(self,other):
        if not isinstance(other, obvec):
            return type(self)((s**other for s in self))
        else:
            raise TypeError("exponents cannot be vectors")
    
    def __rpow__(self,other):
        raise TypeError("exponents cannot be vectors")
    
    @vecoperator
    def __and__(self,other):
        return self&other
    
    def __rand__(self,other):
        self.__and__(other)

    @vecoperator
    def __or__(self,other):
        return self|other
    
    def __ror__(self,other):
        return self.__or__(other)

    def __matmul__(self,other):
        if isinstance(other,obvec):
            if len(self) == len(other):
                res = self[0]*other[0]
                for i in range(1,len(self)):
                    res += (self[i]*other[i])
                return res
            else:
                raise ValueError("operands could not be broadcast together with shapes " +str(len(self))+" != "+str(len(other)))
        else:
            return NotImplemented
    
    def __repr__(self):
        return type(self).__name__ + "([" + ",".join(repr(s) for s in self) + "])"

    def __format__(self,formatstr):
        return type(self).__name__ + "([" + ",".join(s.__format__(formatstr) for s in self) + "])"
    
    #Comparison magic methods
    @vecoperator
    def __eq__(self,other):
        return self==other
    
    def __ne__(self,other):
        return not (self==other)
    
    @vecoperator
    def __lt__(self,other):
        return self<other
    
    @vecoperator
    def __le__(self,other):
        return self<=other
    
    @vecoperator
    def __ge__(self,other):
        return self>=other
    
    @vecoperator
    def __gt__(self,other):
        return self>other
    
    #numpy functions
    def __array__(self,dtype=None,copy=None):
        av = np.zeros(len(self),dtype=type(self[0]))
        for i in range(0,len(self)):
            av[i] = self[i]
        return av
    
    def transpose(self):
        return obmat(([s] for s in self))
    
    T = property(transpose)

def matoperator(func):
    def wrapper(self,other):
        if isinstance(other,obmat):
            if len(self) == len(other):
                return type(self)((func(s,o) for s,o in zip(self,other)))
            else:
                raise ValueError("operands could not be broadcast together with dimensions "+str(len(self[0]))+" x "+str(len(self))+" != "+str(len(other[0]))+" x "+str(len(other)))
        else:
            return type(self)((func(s,other) for s in self))
    return wrapper

class obmat(tuple):
    """A matrix consisting of arbitrary objects.

    An object-matrix is an immutable object that is created by specifying 
    its components column-wise:

        >>> obmat([[O11,O21],[O12,O22],[O13,O23]])
        obmat([obvec([O11,O21]),obvec([O12,O22]),obvec([O13,O23])])

    constructs an object matrix whose entries are references to the objects Oij, where i corresponds to the row and j to the column. 
    Casting into and back from ndarray from the numpy package is supported:

        >>> numpy.array(obmat(numpy.array([[1,4],[7,3]])))
        array([[1, 4],[7, 3]])
    
    All base operations on object-matrix are assigned to the entries in a matrix-like manner. 
    In addition, some matrix operations known from the numpy package, such as matrix product or transpose, are implemented.

        >>> (1 + obmat([[2,4],[7,5]])) / obmat([[1,2],[4,3]]) @ obmat([[1,3],[5,1]])
        obmat([obvec([9.0,8.5]),obvec([17.0,14.5])])

    """

    def __new__(self,args):
        om = tuple.__new__(self,(obvec(c) for c in args))
        rows = len(om[0])
        #if len(om) == 1:
         #   return obvec(om[0])
        if all(len(v) == rows for v in om):
            return om.T if isinstance(args,np.ndarray) else om
        else:
            raise TypeError("expected matrix dimension not met " + str(len(om[0])) +" x "+ str(len(om)))
            
    #Unary operators and functions
    def __pos__(self):
        return self
    
    def __neg__(self):
        return type(self)((-s for s in self))
    
    def __abs__(self):
        return type(self)((s.abs() for s in self))
    
    def __invert__(self):
        return type(self)((s.invert() for s in self))
    
    def __round__(self,n):
        return type(self)((s.round(n) for s in self))
    
    #Normal/Reflected arithmetic operators
    @matoperator
    def __add__(self, other):
        return self+other
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self,other):
        return self.__add__(other.__neg__())
    
    def __rsub__(self,other):
        return (self.__neg__()).__add__(other)
    
    @matoperator
    def __mul__(self,other):
        return self*other
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    @matoperator
    def __div__(self,other):
        return self/other
    
    __truediv__ = __div__
    
    def __rdiv__(self,other):
        return self.__div__(other)
    
    __rtruediv__ = __rdiv__

    @matoperator
    def __mod__(self,other):
        return self%other
    
    def __rmod__(self,other):
        return self.__mod__(other)
        
    def __pow__(self,other):
        if not isinstance(other, obmat):
            return type(self)((s**other for s in self))
        else:
            raise TypeError("exponents cannot be matrices")
    
    def __rpow__(self,other):
        raise TypeError("exponents cannot be matrices")
    
    @matoperator
    def __and__(self,other):
        return self&other
    
    def __rand__(self,other):
        self.__and__(other)

    @matoperator
    def __or__(self,other):
        return self|other
    
    def __ror__(self,other):
        return self.__or__(other)

    def __matmul__(self,other):
        if isinstance(other,obmat):
            if len(self) == len(other[0]):
                return type(self)(((r[u] for r in ([obvec((s[i] for s in self))@o for o in other] for i in range(0,len(self[0])))) for u in range(0,len(other))))
            else:
                raise ValueError("operands could not be broadcast together with dimensions "+str(len(self[0]))+" x "+str(len(self))+" != "+str(len(other[0]))+" x "+str(len(other)))
        elif isinstance(other,obvec):
            if len(self) == len(other):
                return type(other)(((obvec((s[i] for s in self))@other) for i in range(0,len(self[0]))))
            else:
                raise ValueError("operands could not be broadcast together with dimensions "+str(len(self[0]))+" x "+str(len(self))+" != "+str(len(other))+" x 1")
        else:
            return NotImplemented
        
    def __rmatmul__(self,other):
        if isinstance(other,obvec):
            if len(self[0]) == 1:
                return type(self)(((r[u] for r in ([s[0]*other[i] for s in self] for i in range(0,len(other))))) for u in range(0,len(self)))
            else:
                raise ValueError("operands could not be broadcast together with dimensions "+str(len(other))+" x 1"+" != "+str(len(self[0]))+" x "+str(len(self)))
        else:
            return NotImplemented
    
    def __repr__(self):
        return type(self).__name__ + "([" + ",".join(repr(s) for s in self) + "])"

    def __format__(self,formatstr):
        return type(self).__name__ + "([" + ",".join(s.__format__(formatstr) for s in self) + "])"
    
    #Comparison magic methods
    @matoperator
    def __eq__(self,other):
        return self==other
    
    def __ne__(self,other):
        return not (self==other)
    
    @matoperator
    def __lt__(self,other):
        return self<other
    
    @matoperator
    def __le__(self,other):
        return self<=other
    
    @matoperator
    def __ge__(self,other):
        return self>=other
    
    @matoperator
    def __gt__(self,other):
        return self>other
    
    #numpy functions
    def __array__(self,dtype=None,copy=None):
        am = np.zeros([len(self[0]),len(self)],dtype=type(self[0][0]))
        for i in range(0,len(self[0])):
            for j in range(0,len(self)):
                am[i][j] = self[j][i]
        return am
    
    def transpose(self):
        return type(self)(((s[i] for s in self) for i in range(0,len(self[0]))))
    
    T = property(transpose)

class interval(interval):
    @property
    def width(self):
        """The interval consisting only of the width of each component."""
        return self.new(self.Component(x,x) for x in (c.sup - c.inf for c in self))

class intvec(obvec):
    """A vector consisting of intervals from the pyinterval package.
    
    Providing some helper functions to get the width/ midpoint or to split the representative box of the interval vector.

        >>> intvec([[1,3],[2,5]]).split()
        (intvec([interval([1.0, 3.0]),interval([2.0, 3.5])]),
        intvec([interval([1.0, 3.0]),interval([3.5, 5.0])]))
        
    """

    def __new__(self,args):
        return obvec.__new__(self,(interval(a) for a in args))
    
    def width(self):
        '''Returns the width of an interval-vector according to max_{1≤i≤n} w(X_i) as float.'''
        return max(s.width[0][0] for s in self)

    def midpoint(self):
        '''Returns the midpoint of an interval-vector according to (x.sup - x.inf)/2 as numpy array.'''
        return obvec((s.midpoint[0][0] for s in self))
    
    def split(self):
        '''Returns two interval-vectors by splitting the interval with the greatest width (argmax_{1≤i≤n} w(X_i)) of the argument along his midpoint.'''
        X1 = list(self)
        X2 = list(self)
        maxi = max(range(len(self)), key=lambda i: self[i].width[0][0]) #bei gleicher width wird firstindex gewählt
        #maxw, maxi = max((Xi.width[0][0],i) for i, Xi in enumerate(X)) #bei gleicher width wird maxindex gewählt
        X1[maxi] = interval([self[maxi][0].inf, self[maxi].midpoint[0][0]])
        X2[maxi] = interval([self[maxi].midpoint[0][0], self[maxi][0].sup])
        return intvec(X1), intvec(X2)
    
    @property
    def inf(self):
        return obvec(s[0].inf for s in self)
    
    @property
    def sup(self):
        return obvec(s[-1].sup for s in self)

#helper functions
#-------------------------------------------------------------------------------------------------------------
def exp(x):
    """Refers to the corresponding mathematical function matching the input type."""
    return imath.exp(x) if isinstance(x,interval) else np.exp(x)

def log(x):
    """Refers to the corresponding mathematical function matching the input type."""
    return imath.log(x) if isinstance(x,interval) else np.log(x)

def sin(x):
    """Refers to the corresponding mathematical function matching the input type."""
    return imath.sin(x) if isinstance(x,interval) else np.sin(x)

def cos(x):
    """Refers to the corresponding mathematical function matching the input type."""
    return imath.cos(x) if isinstance(x,interval) else np.cos(x)

def tan(x):
    """Refers to the corresponding mathematical function matching the input type."""
    return imath.tan(x) if isinstance(x,interval) else np.tan(x)

def sqrt(x):
    """Refers to the corresponding mathematical function matching the input type."""
    return imath.sqrt(x) if isinstance(x,interval) else np.sqrt(x)