# Analyzer
import numpy as np
from typing import Union, List, Tuple, Any
import sys

class DataAnalytics:
    """
    A comprehensive NumPy-based data analytics toolkit that integrates 
    NumPy functionalities with Object-Oriented Programming principles.
    """
    
    def _init_(self, data: Union[list, np.ndarray, None] = None):
        """
        Initialize the DataAnalytics class with optional data.
        
        Args:
            data: Initial data to work with (list or numpy array)
        """
        if data is not None:
            self.data = np.array(data)
        else:
            self.data = None
    
    # ==================== Array Management ====================
    
    def create_array(self, shape: Tuple[int, ...], fill_value: float = 0.0) -> np.ndarray:
        """Create a new array with specified shape and fill value."""
        self.data = np.full(shape, fill_value)
        return self.data
    
    def create_random_array(self, shape: Tuple[int, ...], min_val: float = 0, max_val: float = 1) -> np.ndarray:
        """Create a random array with specified shape and value range."""
        self.data = np.random.uniform(min_val, max_val, shape)
        return self.data
    
    def get_element(self, *indices) -> Any:
        """Access specific elements using indexing."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return self.data[indices]
    
    def get_slice(self, slice_obj: Union[slice, Tuple[slice, ...]]) -> np.ndarray:
        """Get a slice of the array."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return self.data[slice_obj]
    
    def combine_arrays(self, other_array: np.ndarray, axis: int = 0) -> np.ndarray:
        """Concatenate current array with another array."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        self.data = np.concatenate([self.data, other_array], axis=axis)
        return self.data
    
    def split_array(self, indices: List[int], axis: int = 0) -> List[np.ndarray]:
        """Split array into smaller arrays at specified indices."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.split(self.data, indices, axis=axis)
    
    # ==================== Mathematical Operations ====================
    
    def element_wise_add(self, value: Union[float, np.ndarray]) -> np.ndarray:
        """Perform element-wise addition."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        self.data = self.data + value
        return self.data
    
    def element_wise_subtract(self, value: Union[float, np.ndarray]) -> np.ndarray:
        """Perform element-wise subtraction."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        self.data = self.data - value
        return self.data
    
    def element_wise_multiply(self, value: Union[float, np.ndarray]) -> np.ndarray:
        """Perform element-wise multiplication."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        self.data = self.data * value
        return self.data
    
    def element_wise_divide(self, value: Union[float, np.ndarray]) -> np.ndarray:
        """Perform element-wise division."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        self.data = self.data / value
        return self.data
    
    def dot_product(self, other_array: np.ndarray) -> np.ndarray:
        """Calculate dot product for 2D arrays."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.dot(self.data, other_array)
    
    def matrix_multiplication(self, other_array: np.ndarray) -> np.ndarray:
        """Calculate matrix multiplication for 2D arrays."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.matmul(self.data, other_array)
    
    # ==================== Search, Sort, and Filter ====================
    
    def search_value(self, value: float) -> np.ndarray:
        """Find indices where specific value occurs."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.where(self.data == value)
    
    def sort_array(self, ascending: bool = True, axis: int = -1) -> np.ndarray:
        """Sort array in ascending or descending order."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        if ascending:
            self.data = np.sort(self.data, axis=axis)
        else:
            self.data = np.sort(self.data, axis=axis)[::-1]
        return self.data
    
    def filter_array(self, condition_func) -> np.ndarray:
        """Filter array based on user-defined condition."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return self.data[condition_func(self.data)]
    
    # ==================== Aggregating Functions ====================
    
    def compute_sum(self, axis: int = None) -> Union[float, np.ndarray]:
        """Calculate sum of array elements."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.sum(self.data, axis=axis)
    
    def compute_mean(self, axis: int = None) -> Union[float, np.ndarray]:
        """Calculate mean of array elements."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.mean(self.data, axis=axis)
    
    def compute_median(self, axis: int = None) -> Union[float, np.ndarray]:
        """Calculate median of array elements."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.median(self.data, axis=axis)
    
    def compute_std(self, axis: int = None) -> Union[float, np.ndarray]:
        """Calculate standard deviation of array elements."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.std(self.data, axis=axis)
    
    def compute_variance(self, axis: int = None) -> Union[float, np.ndarray]:
        """Calculate variance of array elements."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.var(self.data, axis=axis)
    
    # ==================== Statistical Functions ====================
    
    def get_min(self, axis: int = None) -> Union[float, np.ndarray]:
        """Get minimum value(s)."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.min(self.data, axis=axis)
    
    def get_max(self, axis: int = None) -> Union[float, np.ndarray]:
        """Get maximum value(s)."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.max(self.data, axis=axis)
    
    def get_percentile(self, percentile: float, axis: int = None) -> Union[float, np.ndarray]:
        """Calculate specified percentile."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        return np.percentile(self.data, percentile, axis=axis)
    
    def correlation_coefficient(self, other_array: np.ndarray) -> np.ndarray:
        """Calculate correlation coefficient between arrays."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        if self.data.ndim == 1 and other_array.ndim == 1:
            return np.corrcoef(self.data, other_array)
        else:
            # For 2D arrays, calculate correlation between columns
            combined = np.column_stack([self.data.flatten(), other_array.flatten()])
            return np.corrcoef(combined.T)
    
    # ==================== Class and Static Methods ====================
    
    @classmethod
    def from_list(cls, data_list: List[Any]):
        """Class method to create instance from list."""
        return cls(data_list)
    
    @staticmethod
    def generate_sequence(start: float, stop: float, step: float) -> np.ndarray:
        """Static method to generate arithmetic sequence."""
        return np.arange(start, stop, step)
    
    @staticmethod
    def generate_linspace(start: float, stop: float, num: int) -> np.ndarray:
        """Static method to generate linearly spaced array."""
        return np.linspace(start, stop, num)
    
    # ==================== Utility Methods ====================
    
    def get_shape(self) -> Tuple[int, ...]:
        """Get shape of current data."""
        if self.data is None:
            return None
        return self.data.shape
    
    def get_dtype(self) -> np.dtype:
        """Get data type of current data."""
        if self.data is None:
            return None
        return self.data.dtype
    
    def reshape(self, new_shape: Tuple[int, ...]) -> np.ndarray:
        """Reshape the current data."""
        if self.data is None:
            raise ValueError("No data loaded. Please create or load data first.")
        self.data = self.data.reshape(new_shape)
        return self.data
    
    def display_data(self, max_elements: int = 100) -> None:
        """Display current data with optional limit."""
        if self.data is None:
            print("No data loaded.")
            return
        
        if self.data.size <= max_elements:
            print(f"Current data (shape: {self.data.shape}):")
            print(self.data)
        else:
            print(f"Current data (shape: {self.data.shape}, showing first {max_elements} elements):")
            flat_data = self.data.flatten()
            print(flat_data[:max_elements])
            print("...")


class NumPyAnalyzerUI:
    """User Interface class for the NumPy Analyzer."""
    
    def _init_(self):
        self.analyzer = DataAnalytics()
        self.running = True
    
    def display_welcome(self):
        """Display welcome message."""
        print("=" * 40)
        print("Welcome to the NumPy Analyzer!")
        print("=" * 40)
        print("A comprehensive toolkit for data analysis using NumPy")
        print()
    
    def display_menu(self):
        """Display the main menu options."""
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        print("1.  Array Creation and Management")
        print("2.  Mathematical Operations")
        print("3.  Search, Sort, and Filter")
        print("4.  Aggregating Functions")
        print("5.  Statistical Functions")
        print("6.  Array Information and Display")
        print("7.  Advanced Operations")
        print("8.  Exit Program")
        print("=" * 50)
    
    def array_management_menu(self):
        """Handle array creation and management operations."""
        while True:
            print("\n" + "-" * 30)
            print("ARRAY MANAGEMENT")
            print("-" * 30)
            print("1. Create array with shape and fill value")
            print("2. Create random array")
            print("3. Create from list")
            print("4. Get element/slice")
            print("5. Combine arrays")
            print("6. Split array")
            print("7. Back to main menu")
            
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == '1':
                self.create_array()
            elif choice == '2':
                self.create_random_array()
            elif choice == '3':
                self.create_from_list()
            elif choice == '4':
                self.get_element_slice()
            elif choice == '5':
                self.combine_arrays()
            elif choice == '6':
                self.split_array()
            elif choice == '7':
                break
            else:
                print("Invalid choice. Please try again.")
    
    def mathematical_operations_menu(self):
        """Handle mathematical operations."""
        while True:
            print("\n" + "-" * 30)
            print("MATHEMATICAL OPERATIONS")
            print("-" * 30)
            print("1. Element-wise addition")
            print("2. Element-wise subtraction")
            print("3. Element-wise multiplication")
            print("4. Element-wise division")
            print("5. Dot product")
            print("6. Matrix multiplication")
            print("7. Back to main menu")
            
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == '1':
                self.element_wise_operation('add')
            elif choice == '2':
                self.element_wise_operation('subtract')
            elif choice == '3':
                self.element_wise_operation('multiply')
            elif choice == '4':
                self.element_wise_operation('divide')
            elif choice == '5':
                self.dot_product_operation()
            elif choice == '6':
                self.matrix_multiplication_operation()
            elif choice == '7':
                break
            else:
                print("Invalid choice. Please try again.")
    
    def search_sort_filter_menu(self):
        """Handle search, sort, and filter operations."""
        while True:
            print("\n" + "-" * 30)
            print("SEARCH, SORT, AND FILTER")
            print("-" * 30)
            print("1. Search for value")
            print("2. Sort array")
            print("3. Filter array")
            print("4. Back to main menu")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                self.search_value()
            elif choice == '2':
                self.sort_array()
            elif choice == '3':
                self.filter_array()
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please try again.")
    
    def aggregating_functions_menu(self):
        """Handle aggregating functions."""
        while True:
            print("\n" + "-" * 30)
            print("AGGREGATING FUNCTIONS")
            print("-" * 30)
            print("1. Sum")
            print("2. Mean")
            print("3. Median")
            print("4. Standard Deviation")
            print("5. Variance")
            print("6. Back to main menu")
            
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                self.compute_aggregation('sum')
            elif choice == '2':
                self.compute_aggregation('mean')
            elif choice == '3':
                self.compute_aggregation('median')
            elif choice == '4':
                self.compute_aggregation('std')
            elif choice == '5':
                self.compute_aggregation('variance')
            elif choice == '6':
                break
            else:
                print("Invalid choice. Please try again.")
    
    def statistical_functions_menu(self):
        """Handle statistical functions."""
        while True:
            print("\n" + "-" * 30)
            print("STATISTICAL FUNCTIONS")
            print("-" * 30)
            print("1. Minimum value")
            print("2. Maximum value")
            print("3. Percentile")
            print("4. Correlation coefficient")
            print("5. Back to main menu")
            
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                self.get_min_max('min')
            elif choice == '2':
                self.get_min_max('max')
            elif choice == '3':
                self.get_percentile()
            elif choice == '4':
                self.correlation_coefficient()
            elif choice == '5':
                break
            else:
                print("Invalid choice. Please try again.")
    
    def info_display_menu(self):
        """Handle array information and display."""
        while True:
            print("\n" + "-" * 30)
            print("ARRAY INFORMATION AND DISPLAY")
            print("-" * 30)
            print("1. Display current data")
            print("2. Get array shape")
            print("3. Get data type")
            print("4. Reshape array")
            print("5. Back to main menu")
            
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                self.display_current_data()
            elif choice == '2':
                self.get_shape()
            elif choice == '3':
                self.get_dtype()
            elif choice == '4':
                self.reshape_array()
            elif choice == '5':
                break
            else:
                print("Invalid choice. Please try again.")
    
    def advanced_operations_menu(self):
        """Handle advanced operations."""
        while True:
            print("\n" + "-" * 30)
            print("ADVANCED OPERATIONS")
            print("-" * 30)
            print("1. Generate arithmetic sequence")
            print("2. Generate linearly spaced array")
            print("3. Create from class method")
            print("4. Back to main menu")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                self.generate_sequence()
            elif choice == '2':
                self.generate_linspace()
            elif choice == '3':
                self.create_from_class_method()
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please try again.")
    
    # Implementation methods for menu options
    
    def create_array(self):
        """Create array with specified shape and fill value."""
        try:
            print("Enter array dimensions (e.g., '3,4' for 3x4 array):")
            dims = input().strip().split(',')
            shape = tuple(int(dim.strip()) for dim in dims)
            fill_value = float(input("Enter fill value (default 0.0): ") or "0.0")
            
            result = self.analyzer.create_array(shape, fill_value)
            print(f"Created array with shape {shape}:")
            print(result)
            
        except Exception as e:
            print(f"Error creating array: {e}")
    
    def create_random_array(self):
        """Create random array."""
        try:
            print("Enter array dimensions (e.g., '3,4' for 3x4 array):")
            dims = input().strip().split(',')
            shape = tuple(int(dim.strip()) for dim in dims)
            min_val = float(input("Enter minimum value (default 0): ") or "0")
            max_val = float(input("Enter maximum value (default 1): ") or "1")
            
            result = self.analyzer.create_random_array(shape, min_val, max_val)
            print(f"Created random array with shape {shape}:")
            print(result)
            
        except Exception as e:
            print(f"Error creating random array: {e}")
    
    def create_from_list(self):
        """Create array from user input list."""
        try:
            print("Enter comma-separated values (e.g., '1,2,3,4'):")
            values = input().strip().split(',')
            data_list = [float(val.strip()) for val in values]
            
            self.analyzer = DataAnalytics(data_list)
            print("Created array from list:")
            print(self.analyzer.data)
            
        except Exception as e:
            print(f"Error creating array from list: {e}")
    
    def get_element_slice(self):
        """Get element or slice from array."""
        try:
            if self.analyzer.data is None:
                print("No data loaded. Please create an array first.")
                return
            
            print("Current array shape:", self.analyzer.get_shape())
            print("Enter indices (e.g., '1,2' for element at [1,2], or '1:3,0:2' for slice):")
            indices_str = input().strip()
            
            # Simple parsing for demonstration
            if ':' in indices_str:
                print("Slice operation - displaying current data:")
                self.analyzer.display_data()
            else:
                indices = tuple(int(idx.strip()) for idx in indices_str.split(','))
                element = self.analyzer.get_element(*indices)
                print(f"Element at {indices}: {element}")
                
        except Exception as e:
            print(f"Error getting element/slice: {e}")
    
    def element_wise_operation(self, operation):
        """Perform element-wise operations."""
        try:
            if self.analyzer.data is None:
                print("No data loaded. Please create an array first.")
                return
            
            value = float(input(f"Enter value for {operation}: "))
            
            if operation == 'add':
                result = self.analyzer.element_wise_add(value)
            elif operation == 'subtract':
                result = self.analyzer.element_wise_subtract(value)
            elif operation == 'multiply':
                result = self.analyzer.element_wise_multiply(value)
            elif operation == 'divide':
                result = self.analyzer.element_wise_divide(value)
            
            print(f"Result after {operation}:")
            print(result)
            
        except Exception as e:
            print(f"Error performing {operation}: {e}")
    
    def compute_aggregation(self, func_name):
        """Compute aggregation functions."""
        try:
            if self.analyzer.data is None:
                print("No data loaded. Please create an array first.")
                return
            
            if func_name == 'sum':
                result = self.analyzer.compute_sum()
            elif func_name == 'mean':
                result = self.analyzer.compute_mean()
            elif func_name == 'median':
                result = self.analyzer.compute_median()
            elif func_name == 'std':
                result = self.analyzer.compute_std()
            elif func_name == 'variance':
                result = self.analyzer.compute_variance()
            
            print(f"{func_name.capitalize()}: {result}")
            
        except Exception as e:
            print(f"Error computing {func_name}: {e}")
    
    def display_current_data(self):
        """Display current array data."""
        self.analyzer.display_data()
    
    def get_shape(self):
        """Get and display array shape."""
        shape = self.analyzer.get_shape()
        if shape:
            print(f"Array shape: {shape}")
        else:
            print("No data loaded.")
    
    def get_dtype(self):
        """Get and display array data type."""
        dtype = self.analyzer.get_dtype()
        if dtype:
            print(f"Array data type: {dtype}")
        else:
            print("No data loaded.")
    
    def run(self):
        """Main program loop."""
        self.display_welcome()
        
        while self.running:
            self.display_menu()
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1':
                self.array_management_menu()
            elif choice == '2':
                self.mathematical_operations_menu()
            elif choice == '3':
                self.search_sort_filter_menu()
            elif choice == '4':
                self.aggregating_functions_menu()
            elif choice == '5':
                self.statistical_functions_menu()
            elif choice == '6':
                self.info_display_menu()
            elif choice == '7':
                self.advanced_operations_menu()
            elif choice == '8':
                self.exit_program()
            else:
                print("Invalid choice. Please enter a number between 1 and 8.")
    
    def exit_program(self):
        """Exit the program."""
        print("\nThank you for using the NumPy Analyzer!")
        print("=" * 40)
        self.running = False


# Additional stub implementations for remaining methods
    def combine_arrays(self):
        """Stub for combine arrays functionality."""
        print("Combine arrays functionality - would allow concatenating with another array")
    
    def split_array(self):
        """Stub for split array functionality."""
        print("Split array functionality - would allow splitting array into parts")
    
    def search_value(self):
        """Stub for search value functionality."""
        print("Search value functionality - would find indices of specific values")
    
    def sort_array(self):
        """Stub for sort array functionality."""
        print("Sort array functionality - would sort array in ascending/descending order")
    
    def filter_array(self):
        """Stub for filter array functionality."""
        print("Filter array functionality - would filter based on conditions")
    
    def get_min_max(self, operation):
        """Stub for min/max operations."""
        print(f"Get {operation} functionality - would find {operation} values")
    
    def get_percentile(self):
        """Stub for percentile calculation."""
        print("Percentile functionality - would calculate specified percentiles")
    
    def correlation_coefficient(self):
        """Stub for correlation coefficient."""
        print("Correlation coefficient functionality - would calculate correlation between arrays")
    
    def reshape_array(self):
        """Stub for reshape functionality."""
        print("Reshape functionality - would change array dimensions")
    
    def generate_sequence(self):
        """Stub for sequence generation."""
        print("Generate sequence functionality - would create arithmetic sequences")
    
    def generate_linspace(self):
        """Stub for linspace generation."""
        print("Generate linspace functionality - would create linearly spaced arrays")
    
    def create_from_class_method(self):
        """Stub for class method creation."""
        print("Class method functionality - would create instances using class methods")
    
    def dot_product_operation(self):
        """Stub for dot product."""
        print("Dot product functionality - would calculate dot products of arrays")
    
    def matrix_multiplication_operation(self):
        """Stub for matrix multiplication."""
        print("Matrix multiplication functionality - would perform matrix operations")


if _name_ == "_main_":
    # Run the NumPy Analyzer
    analyzer_ui = NumPyAnalyzerUI()
    analyzer_ui.run()
