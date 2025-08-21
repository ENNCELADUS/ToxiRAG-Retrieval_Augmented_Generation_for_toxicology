"""
Unit tests for data normalization utilities.
"""

import pytest
from ingest.normalization import (
    TumorVolumeNormalizer, WeightNormalizer, OrganMassNormalizer, 
    DoseNormalizer, TimelineNormalizer, StrainNormalizer, SexNormalizer,
    DataNormalizer
)


class TestTumorVolumeNormalizer:
    """Test tumor volume normalization."""
    
    def setup_method(self):
        self.normalizer = TumorVolumeNormalizer()
    
    def test_normalize_mm3(self):
        """Test mm³ (no conversion needed)."""
        result = self.normalizer.normalize_volume("150.5 mm³")
        assert result.value == 150.5
        assert result.unit == "mm³"
        assert result.conversion_applied is None
    
    def test_normalize_cm3(self):
        """Test cm³ to mm³ conversion."""
        result = self.normalizer.normalize_volume("2.5 cm³")
        assert result.value == 2500.0
        assert result.unit == "mm³"
        assert "cm3 → mm³" in result.conversion_applied  # Note: implementation normalizes superscript
    
    def test_normalize_ml(self):
        """Test mL to mm³ conversion."""
        result = self.normalizer.normalize_volume("1.5 mL")
        assert result.value == 1500.0
        assert result.unit == "mm³"
        assert "mL → mm³" in result.conversion_applied
    
    def test_unknown_value(self):
        """Test unknown values."""
        result = self.normalizer.normalize_volume("未说明")
        assert result.value is None
        assert result.unit == "mm³"
    
    def test_invalid_format(self):
        """Test invalid format."""
        result = self.normalizer.normalize_volume("invalid volume")
        assert result.value is None


class TestWeightNormalizer:
    """Test weight normalization."""
    
    def setup_method(self):
        self.normalizer = WeightNormalizer()
    
    def test_normalize_grams(self):
        """Test grams (no conversion)."""
        result = self.normalizer.normalize_weight("25g")
        assert result.value == 25.0
        assert result.unit == "g"
        assert result.conversion_applied is None
    
    def test_normalize_kg(self):
        """Test kg to g conversion."""
        result = self.normalizer.normalize_weight("0.025kg")
        assert result.value == 25.0
        assert result.unit == "g"
        assert "kg → g" in result.conversion_applied
    
    def test_normalize_mg(self):
        """Test mg to g conversion."""
        result = self.normalizer.normalize_weight("25000mg")
        assert result.value == 25.0
        assert result.unit == "g"
        assert "mg → g" in result.conversion_applied


class TestOrganMassNormalizer:
    """Test organ mass normalization."""
    
    def setup_method(self):
        self.normalizer = OrganMassNormalizer()
    
    def test_normalize_mg(self):
        """Test mg (no conversion)."""
        result = self.normalizer.normalize_organ_mass("150mg")
        assert result.value == 150.0
        assert result.unit == "mg"
        assert result.conversion_applied is None
    
    def test_normalize_g(self):
        """Test g to mg conversion."""
        result = self.normalizer.normalize_organ_mass("0.15g")
        assert result.value == 150.0
        assert result.unit == "mg"
        assert "g → mg" in result.conversion_applied


class TestDoseNormalizer:
    """Test dose normalization."""
    
    def setup_method(self):
        self.normalizer = DoseNormalizer()
    
    def test_normalize_daily_dose(self):
        """Test daily dose."""
        dose, freq, daily = self.normalizer.normalize_dose("200 mg/kg daily")
        assert dose == 200.0
        assert freq == "qd"
        assert daily == 200.0
    
    def test_normalize_bid_dose(self):
        """Test twice daily dose."""
        dose, freq, daily = self.normalizer.normalize_dose("100 mg/kg bid")
        assert dose == 100.0
        assert freq == "bid"
        assert daily == 200.0
    
    def test_normalize_q2d_dose(self):
        """Test every other day dose."""
        dose, freq, daily = self.normalizer.normalize_dose("400 mg/kg 隔日")
        assert dose == 400.0
        assert freq == "q2d"
        assert daily == 200.0
    
    def test_normalize_weekly_dose(self):
        """Test weekly dose."""
        dose, freq, daily = self.normalizer.normalize_dose("1400 mg/kg weekly")
        assert dose == 1400.0
        assert freq == "qwk"
        assert daily == 200.0
    
    def test_unknown_dose(self):
        """Test unknown dose."""
        dose, freq, daily = self.normalizer.normalize_dose("未说明")
        assert dose is None
        assert freq == "未说明"
        assert daily is None


class TestTimelineNormalizer:
    """Test timeline normalization."""
    
    def setup_method(self):
        self.normalizer = TimelineNormalizer()
    
    def test_normalize_day(self):
        """Test Day N format."""
        result = self.normalizer.normalize_timepoint("Day 7")
        assert result == 7
    
    def test_normalize_day_plus(self):
        """Test Day N+M format."""
        result = self.normalizer.normalize_timepoint("Day 7+3")
        assert result == 10
    
    def test_normalize_week(self):
        """Test Week N format."""
        result = self.normalizer.normalize_timepoint("Week 2")
        assert result == 14
    
    def test_normalize_chinese_days(self):
        """Test Chinese day format."""
        result = self.normalizer.normalize_timepoint("14天")
        assert result == 14
    
    def test_unknown_timepoint(self):
        """Test unknown timepoint."""
        result = self.normalizer.normalize_timepoint("未说明")
        assert result is None


class TestStrainNormalizer:
    """Test strain normalization."""
    
    def setup_method(self):
        self.normalizer = StrainNormalizer()
    
    def test_normalize_c57bl6(self):
        """Test C57BL/6 variants."""
        raw, norm = self.normalizer.normalize_strain("C57BL/6J")
        assert raw == "C57BL/6J"
        assert norm == "C57BL/6"
        
        raw, norm = self.normalizer.normalize_strain("C57BL/6N")
        assert norm == "C57BL/6"
    
    def test_normalize_balbc(self):
        """Test BALB/c variants."""
        raw, norm = self.normalizer.normalize_strain("BALB/cJ")
        assert norm == "BALB/c"
    
    def test_normalize_km(self):
        """Test KM strain."""
        raw, norm = self.normalizer.normalize_strain("昆明小鼠")
        assert norm == "KM"
    
    def test_normalize_sd(self):
        """Test SD strain."""
        raw, norm = self.normalizer.normalize_strain("Sprague-Dawley")
        assert norm == "SD"
    
    def test_unknown_strain(self):
        """Test unknown strain."""
        raw, norm = self.normalizer.normalize_strain("Unknown strain")
        assert raw == "Unknown strain"
        assert norm is None


class TestSexNormalizer:
    """Test sex normalization."""
    
    def setup_method(self):
        self.normalizer = SexNormalizer()
    
    def test_normalize_male(self):
        """Test male indicators."""
        assert self.normalizer.normalize_sex("male") == "male"
        assert self.normalizer.normalize_sex("雄性") == "male"
        assert self.normalizer.normalize_sex("公鼠") == "male"
    
    def test_normalize_female(self):
        """Test female indicators."""
        assert self.normalizer.normalize_sex("female") == "female"
        assert self.normalizer.normalize_sex("雌性") == "female"
        assert self.normalizer.normalize_sex("母鼠") == "female"
    
    def test_normalize_mixed(self):
        """Test mixed indicators."""
        assert self.normalizer.normalize_sex("雌雄各半") == "mixed"
        assert self.normalizer.normalize_sex("mixed") == "mixed"
        assert self.normalizer.normalize_sex("male and female") == "mixed"
    
    def test_unknown_sex(self):
        """Test unknown sex."""
        assert self.normalizer.normalize_sex("未说明") == "未说明"
        assert self.normalizer.normalize_sex("") == "未说明"


class TestDataNormalizer:
    """Test the main data normalizer."""
    
    def setup_method(self):
        self.normalizer = DataNormalizer()
    
    def test_normalize_all_fields(self):
        """Test normalizing all fields at once."""
        data = {
            "tumor_volume": "2.5 cm³",
            "body_weight": "0.025kg",
            "organ_mass": "0.15g",
            "dose": "200 mg/kg daily",
            "timepoint": "Day 7",
            "strain": "C57BL/6J",
            "sex": "雌雄各半"
        }
        
        normalized = self.normalizer.normalize_all_fields(data)
        
        # Check that normalized fields are added
        assert normalized["tumor_volume_normalized"].value == 2500.0
        assert normalized["body_weight_normalized"].value == 25.0
        assert normalized["organ_mass_normalized"].value == 150.0
        assert normalized["dose_mg_per_kg"] == 200.0
        assert normalized["dose_frequency_norm"] == "qd"
        assert normalized["canonical_day"] == 7
        assert normalized["strain_norm"] == "C57BL/6"
        assert normalized["sex_norm"] == "mixed"
        assert normalized["units_version"] == "v1.0"
