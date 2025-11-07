import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.extract import DataExtractor
from data.transform import DataTransformer
from data.validate import DataValidator
from utils.config_loader import ConfigLoader


class TestDataQuality:
    """Test suite for data quality checks."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'Patient_ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'Age': [25, 35, 45, 55, 65],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Other'],
            'Height_cm': [175, 165, 180, 160, 170],
            'Weight_kg': [70, 60, 85, 55, 75],
            'BMI': [22.9, 22.0, 26.2, 21.5, 25.9],
            'Chronic_Disease': ['None', 'Diabetes', 'None', 'Hypertension', 'None'],
            'Blood_Pressure_Systolic': [120, 140, 110, 160, 130],
            'Blood_Pressure_Diastolic': [80, 90, 70, 100, 85],
            'Cholesterol_Level': [180, 220, 190, 250, 200],
            'Blood_Sugar_Level': [90, 150, 85, 95, 100],
            'Genetic_Risk_Factor': ['No', 'Yes', 'No', 'Yes', 'No'],
            'Allergies': ['None', 'Nut Allergy', 'None', 'Gluten Intolerance', 'None'],
            'Daily_Steps': [8000, 6000, 10000, 5000, 7500],
            'Exercise_Frequency': [3, 2, 5, 1, 4],
            'Sleep_Hours': [7.5, 6.0, 8.0, 5.5, 7.0],
            'Alcohol_Consumption': ['No', 'Yes', 'No', 'No', 'Yes'],
            'Smoking_Habit': ['No', 'No', 'Yes', 'No', 'No'],
            'Dietary_Habits': ['Regular', 'Vegetarian', 'Regular', 'Vegan', 'Keto'],
            'Caloric_Intake': [2000, 1800, 2500, 1600, 2200],
            'Protein_Intake': [100, 80, 120, 60, 110],
            'Carbohydrate_Intake': [250, 200, 300, 150, 100],
            'Fat_Intake': [80, 70, 90, 50, 150],
            'Preferred_Cuisine': ['Western', 'Mediterranean', 'Asian', 'Indian', 'Western'],
            'Food_Aversions': ['None', 'Spicy', 'None', 'Sweet', 'Salty'],
            'Recommended_Meal_Plan': ['Balanced Diet', 'High-Protein Diet', 'Low-Fat Diet', 'Low-Carb Diet', 'High-Protein Diet']
        })
    
    @pytest.fixture
    def config_loader(self):
        """Create config loader for testing."""
        return ConfigLoader()
    
    @pytest.fixture
    def data_validator(self, config_loader):
        """Create data validator for testing."""
        return DataValidator(config_loader)
    
    def test_data_schema_validation(self, sample_data, data_validator):
        """Test data schema validation."""
        validation_result = data_validator.validate_schema(sample_data)
        
        assert isinstance(validation_result, dict)
        assert 'schema_valid' in validation_result
        assert 'issues' in validation_result
        assert 'warnings' in validation_result
        
        # Should pass with sample data
        assert validation_result['schema_valid'] == True
    
    def test_data_quality_metrics(self, sample_data, data_validator):
        """Test data quality metrics calculation."""
        quality_result = data_validator.validate_data_quality(sample_data)
        
        assert isinstance(quality_result, dict)
        assert 'quality_score' in quality_result
        assert 'metrics' in quality_result
        assert 'issues' in quality_result
        
        # Quality score should be between 0 and 100
        assert 0 <= quality_result['quality_score'] <= 100
        
        # Metrics should contain expected keys
        metrics = quality_result['metrics']
        expected_metrics = ['total_rows', 'total_columns', 'missing_percentage', 'duplicate_percentage']
        for metric in expected_metrics:
            assert metric in metrics
    
    def test_target_distribution_validation(self, sample_data, data_validator):
        """Test target variable distribution validation."""
        target_result = data_validator.validate_target_distribution(sample_data, 'Recommended_Meal_Plan')
        
        assert isinstance(target_result, dict)
        assert 'unique_values' in target_result
        assert 'value_counts' in target_result
        assert 'distribution_balance' in target_result
        
        # Should have multiple unique values
        assert target_result['unique_values'] > 1
    
    def test_feature_distribution_validation(self, sample_data, data_validator):
        """Test feature distribution validation."""
        feature_result = data_validator.validate_feature_distributions(sample_data)
        
        assert isinstance(feature_result, dict)
        assert 'numeric_features' in feature_result
        assert 'categorical_features' in feature_result
        assert 'issues' in feature_result
    
    def test_missing_values_detection(self, data_validator):
        """Test missing values detection."""
        # Create data with missing values
        data_with_missing = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': ['a', None, 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, None, 5.5]
        })
        
        quality_result = data_validator.validate_data_quality(data_with_missing)
        
        # Should detect missing values
        assert quality_result['metrics']['missing_percentage'] > 0
    
    def test_duplicate_detection(self, data_validator):
        """Test duplicate rows detection."""
        # Create data with duplicates
        data_with_duplicates = pd.DataFrame({
            'A': [1, 2, 2, 4, 5],
            'B': ['a', 'b', 'b', 'd', 'e'],
            'C': [1.1, 2.2, 2.2, 4.4, 5.5]
        })
        
        quality_result = data_validator.validate_data_quality(data_with_duplicates)
        
        # Should detect duplicates
        assert quality_result['metrics']['duplicate_percentage'] > 0
    
    def test_outlier_detection(self, data_validator):
        """Test outlier detection in numeric columns."""
        # Create data with outliers
        normal_data = np.random.normal(50, 10, 100)
        outlier_data = np.append(normal_data, [200, -50])  # Add outliers
        
        data_with_outliers = pd.DataFrame({
            'normal_col': normal_data.tolist() + [50, 50],
            'outlier_col': outlier_data
        })
        
        feature_result = data_validator.validate_feature_distributions(data_with_outliers)
        
        # Should have numeric features analysis
        assert len(feature_result['numeric_features']) > 0
    
    def test_comprehensive_validation_report(self, sample_data, data_validator):
        """Test comprehensive validation report generation."""
        report = data_validator.generate_validation_report(sample_data, 'Recommended_Meal_Plan')
        
        assert isinstance(report, dict)
        
        # Check required sections
        required_sections = [
            'timestamp', 'dataset_shape', 'schema_validation',
            'quality_validation', 'feature_validation', 'target_validation', 'overall_status'
        ]
        
        for section in required_sections:
            assert section in report
        
        # Check overall status structure
        overall_status = report['overall_status']
        assert 'validation_passed' in overall_status
        assert 'total_issues' in overall_status
        assert 'all_issues' in overall_status


class TestDataTransformation:
    """Test suite for data transformation functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'missing_col': [1, None, 3, None, 5],
            'target_col': ['Class1', 'Class2', 'Class1', 'Class3', 'Class2']
        })
    
    @pytest.fixture
    def config_loader(self):
        """Create config loader for testing."""
        return ConfigLoader()
    
    @pytest.fixture
    def data_transformer(self, config_loader):
        """Create data transformer for testing."""
        return DataTransformer(config_loader)
    
    def test_missing_values_handling(self, sample_data, data_transformer):
        """Test missing values handling."""
        # Mock the config to include our test columns
        data_transformer.data_config = {
            'data': {
                'validation': {
                    'numeric_columns': ['numeric_col', 'missing_col'],
                    'categorical_columns': ['categorical_col']
                }
            }
        }
        
        result = data_transformer.handle_missing_values(sample_data)
        
        # Should have no missing values after handling
        assert result.isnull().sum().sum() == 0
    
    def test_outlier_removal(self, data_transformer):
        """Test outlier removal."""
        # Create data with clear outliers
        data_with_outliers = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'outlier_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]  # 1000 is outlier
        })
        
        # Mock config
        data_transformer.data_config = {
            'data': {
                'validation': {
                    'numeric_columns': ['normal_col', 'outlier_col']
                }
            }
        }
        
        result = data_transformer.remove_outliers(data_with_outliers, method='iqr', threshold=1.5)
        
        # Should remove the outlier row
        assert len(result) < len(data_with_outliers)
    
    def test_categorical_encoding(self, sample_data, data_transformer):
        """Test categorical variable encoding."""
        # Mock config
        data_transformer.data_config = {
            'data': {
                'validation': {
                    'categorical_columns': ['categorical_col', 'target_col']
                }
            }
        }
        
        result = data_transformer.encode_categorical_variables(sample_data, target_column='target_col')
        
        # Should have encoded categorical variables
        assert 'target_col' in result.columns  # Target should be label encoded
        
        # Check if encoders were stored
        assert len(data_transformer.encoders) > 0
    
    def test_numeric_scaling(self, sample_data, data_transformer):
        """Test numeric feature scaling."""
        result = data_transformer.scale_numeric_features(sample_data, target_column='target_col')
        
        # Numeric columns should be scaled (mean ~0, std ~1)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'target_col']
        
        if len(numeric_cols) > 0:
            # Check if scaler was stored
            assert len(data_transformer.scalers) > 0


class TestDataExtraction:
    """Test suite for data extraction functions."""
    
    @pytest.fixture
    def config_loader(self):
        """Create config loader for testing."""
        return ConfigLoader()
    
    @pytest.fixture
    def data_extractor(self, config_loader):
        """Create data extractor for testing."""
        return DataExtractor(config_loader)
    
    def test_csv_extraction(self, data_extractor, tmp_path):
        """Test CSV data extraction."""
        # Create temporary CSV file
        test_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        
        csv_path = tmp_path / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Extract data
        result = data_extractor.extract_csv_data(str(csv_path))
        
        # Should match original data
        pd.testing.assert_frame_equal(result, test_data)
    
    def test_data_info_generation(self, data_extractor):
        """Test data info generation."""
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 1],  # Has duplicate
            'B': ['a', 'b', None, 'd']  # Has missing value
        })
        
        info = data_extractor.get_data_info(test_data)
        
        # Check info structure
        expected_keys = ['shape', 'columns', 'dtypes', 'missing_values', 'memory_usage', 'duplicate_rows']
        for key in expected_keys:
            assert key in info
        
        # Check values
        assert info['shape'] == (4, 2)
        assert info['duplicate_rows'] == 1
        assert info['missing_values']['B'] == 1
    
    def test_column_validation(self, data_extractor):
        """Test required column validation."""
        # Mock config with required columns
        data_extractor.data_config = {
            'data': {
                'validation': {
                    'required_columns': ['A', 'B', 'C']
                }
            }
        }
        
        # Test with all required columns
        complete_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.1, 2.2, 3.3]
        })
        
        assert data_extractor.validate_required_columns(complete_data) == True
        
        # Test with missing columns
        incomplete_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
            # Missing column 'C'
        })
        
        assert data_extractor.validate_required_columns(incomplete_data) == False


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
