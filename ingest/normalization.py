"""
ToxiRAG Data Normalization Utilities
Normalize units, timepoints, doses, and other data according to the specification.
"""

import re
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class NormalizedValue:
    """Normalized value with metadata."""
    value: Optional[float]
    raw_value: str
    unit: str
    conversion_applied: Optional[str] = None
    calc_method: Optional[str] = None


class TumorVolumeNormalizer:
    """Normalize tumor volume measurements to mm³."""
    
    # Unit conversion factors to mm³
    UNIT_CONVERSIONS = {
        'cm³': 1000.0,
        'cm3': 1000.0,
        'ml': 1000.0,
        'mL': 1000.0,
        'mm³': 1.0,
        'mm3': 1.0
    }
    
    # Common volume calculation formulas
    VOLUME_FORMULAS = {
        'ellipsoid': 'V = π/6 × L × W × H',
        'sphere': 'V = 4/3 × π × r³',
        'lwh_over_2': 'V = (L×W²)/2',
        'lw_over_2': 'V = (L×W)/2',
        'pi_lw_over_6': 'V = π×L×W/6'
    }
    
    def normalize_volume(self, volume_str: str) -> NormalizedValue:
        """Normalize tumor volume to mm³."""
        if not volume_str or volume_str.strip() == "未说明":
            return NormalizedValue(
                value=None,
                raw_value=volume_str,
                unit="mm³"
            )
        
        # Clean the input
        volume_str = volume_str.strip()
        
        # Extract numeric value and unit
        volume_match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z³³]+)', volume_str)
        if not volume_match:
            return NormalizedValue(
                value=None,
                raw_value=volume_str,
                unit="mm³"
            )
        
        numeric_value = float(volume_match.group(1))
        unit_str = volume_match.group(2).replace('³', '3')  # normalize superscript
        
        # Apply conversion
        conversion_factor = self.UNIT_CONVERSIONS.get(unit_str, 1.0)
        normalized_value = numeric_value * conversion_factor
        
        conversion_note = None
        if conversion_factor != 1.0:
            conversion_note = f"{unit_str} → mm³ (×{conversion_factor})"
        
        return NormalizedValue(
            value=normalized_value,
            raw_value=volume_str,
            unit="mm³",
            conversion_applied=conversion_note
        )
    
    def parse_diameter_formula(self, formula_str: str, length: float, width: float, height: Optional[float] = None) -> NormalizedValue:
        """Calculate volume from diameters using specified formula."""
        formula_lower = formula_str.lower()
        
        calc_method = None
        volume = None
        
        if 'l×w²' in formula_lower or 'l*w²' in formula_lower or '(l×w²)/2' in formula_lower:
            volume = (length * width * width) / 2
            calc_method = "V = (L×W²)/2"
        elif 'l×w×h' in formula_lower and height is not None:
            volume = length * width * height
            calc_method = "V = L×W×H"
        elif 'π/6' in formula_lower and 'l×w' in formula_lower:
            volume = (3.14159 * length * width) / 6
            calc_method = "V = π×L×W/6"
        elif '4/3' in formula_lower and 'π' in formula_lower:
            # Assume spherical with radius = width/2
            radius = width / 2
            volume = (4/3) * 3.14159 * (radius ** 3)
            calc_method = "V = 4/3×π×r³"
        
        return NormalizedValue(
            value=volume,
            raw_value=f"L={length}, W={width}" + (f", H={height}" if height else ""),
            unit="mm³",
            calc_method=calc_method
        )


class WeightNormalizer:
    """Normalize weight measurements to grams."""
    
    def normalize_weight(self, weight_str: str) -> NormalizedValue:
        """Normalize weight to grams."""
        if not weight_str or weight_str.strip() == "未说明":
            return NormalizedValue(
                value=None,
                raw_value=weight_str,
                unit="g"
            )
        
        # Extract numeric value and unit
        weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g|mg)?', weight_str.lower())
        if not weight_match:
            return NormalizedValue(
                value=None,
                raw_value=weight_str,
                unit="g"
            )
        
        numeric_value = float(weight_match.group(1))
        unit_str = weight_match.group(2) or "g"  # default to grams
        
        # Convert to grams
        if unit_str == "kg":
            normalized_value = numeric_value * 1000
            conversion_note = "kg → g (×1000)"
        elif unit_str == "mg":
            normalized_value = numeric_value / 1000
            conversion_note = "mg → g (÷1000)"
        else:
            normalized_value = numeric_value
            conversion_note = None
        
        return NormalizedValue(
            value=normalized_value,
            raw_value=weight_str,
            unit="g",
            conversion_applied=conversion_note
        )


class OrganMassNormalizer:
    """Normalize organ mass measurements to milligrams."""
    
    def normalize_organ_mass(self, mass_str: str) -> NormalizedValue:
        """Normalize organ mass to milligrams."""
        if not mass_str or mass_str.strip() == "未说明":
            return NormalizedValue(
                value=None,
                raw_value=mass_str,
                unit="mg"
            )
        
        # Extract numeric value and unit
        mass_match = re.search(r'(\d+(?:\.\d+)?)\s*(g|mg|kg)?', mass_str.lower())
        if not mass_match:
            return NormalizedValue(
                value=None,
                raw_value=mass_str,
                unit="mg"
            )
        
        numeric_value = float(mass_match.group(1))
        unit_str = mass_match.group(2) or "mg"  # default to mg
        
        # Convert to milligrams
        if unit_str == "g":
            normalized_value = numeric_value * 1000
            conversion_note = "g → mg (×1000)"
        elif unit_str == "kg":
            normalized_value = numeric_value * 1000000
            conversion_note = "kg → mg (×1,000,000)"
        else:
            normalized_value = numeric_value
            conversion_note = None
        
        return NormalizedValue(
            value=normalized_value,
            raw_value=mass_str,
            unit="mg",
            conversion_applied=conversion_note
        )


class DoseNormalizer:
    """Normalize dose measurements to mg/kg per administration."""
    
    # Frequency mappings to standard codes
    FREQUENCY_MAPPINGS = {
        "daily": "qd",
        "once daily": "qd",
        "qd": "qd",
        "every day": "qd",
        "每日": "qd",
        "每天": "qd",
        "隔日": "q2d",
        "every other day": "q2d",
        "q2d": "q2d",
        "qod": "qod",
        "每隔一天": "q2d",
        "twice daily": "bid",
        "bid": "bid",
        "bis in die": "bid",
        "每日两次": "bid",
        "一日两次": "bid",
        "三次": "tid",
        "tid": "tid",
        "每日三次": "tid",
        "一日三次": "tid",
        "weekly": "qwk",
        "qwk": "qwk",
        "每周": "qwk",
        "每周一次": "qwk"
    }
    
    def normalize_dose(self, dose_str: str) -> Tuple[Optional[float], Optional[str], Optional[float]]:
        """
        Normalize dose to mg/kg per administration.
        
        Returns:
            (dose_mg_per_kg, frequency_norm, daily_equiv_mg_per_kg)
        """
        if not dose_str or dose_str.strip() == "未说明":
            return None, "未说明", None
        
        dose_str_lower = dose_str.lower()
        
        # Extract dose value (mg/kg)
        dose_match = re.search(r'(\d+(?:\.\d+)?)\s*mg/kg', dose_str_lower)
        if not dose_match:
            return None, "未说明", None
        
        dose_mg_per_kg = float(dose_match.group(1))
        
        # Extract frequency
        frequency_norm = "未说明"
        for pattern, norm in self.FREQUENCY_MAPPINGS.items():
            if pattern in dose_str_lower:
                frequency_norm = norm
                break
        
        # Calculate daily equivalent if frequency is known
        daily_equiv = None
        if frequency_norm == "qd":
            daily_equiv = dose_mg_per_kg
        elif frequency_norm == "bid":
            daily_equiv = dose_mg_per_kg * 2
        elif frequency_norm == "tid":
            daily_equiv = dose_mg_per_kg * 3
        elif frequency_norm == "q2d":
            daily_equiv = dose_mg_per_kg * 0.5
        elif frequency_norm == "qwk":
            daily_equiv = dose_mg_per_kg / 7
        # Only set daily_equiv if explicitly stated in paper, otherwise None
        
        return dose_mg_per_kg, frequency_norm, daily_equiv


class TimelineNormalizer:
    """Normalize timeline points to days since inoculation."""
    
    def normalize_timepoint(self, timepoint_str: str) -> Optional[int]:
        """Normalize timepoint to canonical day number."""
        if not timepoint_str or timepoint_str.strip() == "未说明":
            return None
        
        # Handle "Day N" and "Day N+M" formats
        day_match = re.search(r'day\s*(\d+)(?:\s*\+\s*(\d+))?', timepoint_str.lower())
        if day_match:
            base_day = int(day_match.group(1))
            extra_day = int(day_match.group(2)) if day_match.group(2) else 0
            return base_day + extra_day
        
        # Handle "Week N" format (only if explicitly stated as week equivalent)
        week_match = re.search(r'week\s*(\d+)', timepoint_str.lower())
        if week_match and '天' not in timepoint_str:  # Only if not mixed with days
            weeks = int(week_match.group(1))
            return weeks * 7  # Convert weeks to days
        
        # Handle direct day numbers
        direct_day_match = re.search(r'(\d+)\s*天', timepoint_str)
        if direct_day_match:
            return int(direct_day_match.group(1))
        
        return None


class StrainNormalizer:
    """Normalize animal strain names."""
    
    STRAIN_MAPPINGS = {
        "C57BL/6": "C57BL/6",
        "C57BL/6J": "C57BL/6",
        "C57BL/6N": "C57BL/6",
        "C57Bl/6": "C57BL/6",
        "BALB/c": "BALB/c",
        "BALB/cJ": "BALB/c",
        "BALB/cAnN": "BALB/c",
        "KM": "KM",
        "昆明小鼠": "KM",
        "SD": "SD",
        "Sprague-Dawley": "SD",
        "Sprague Dawley": "SD",
        "SD大鼠": "SD",
        "Wistar": "Wistar",
        "Wistar大鼠": "Wistar"
    }
    
    def normalize_strain(self, strain_str: str) -> Tuple[str, Optional[str]]:
        """
        Normalize strain name.
        
        Returns:
            (raw_strain, normalized_strain)
        """
        if not strain_str or strain_str.strip() == "未说明":
            return "未说明", None
        
        strain_clean = strain_str.strip()
        
        # Find exact or close matches
        for raw_pattern, normalized in self.STRAIN_MAPPINGS.items():
            if raw_pattern.lower() in strain_clean.lower():
                return strain_clean, normalized
        
        # No match found
        return strain_clean, None


class SexNormalizer:
    """Normalize sex/gender information."""
    
    def normalize_sex(self, sex_str: str) -> str:
        """Normalize sex to standard values."""
        if not sex_str or sex_str.strip() == "未说明":
            return "未说明"
        
        sex_lower = sex_str.lower()
        
        # Use exact matches to avoid "female" containing "male"
        male_terms = ['雄', '公', '雄性']
        female_terms = ['雌', '母', '雌性']
        
        # Check for exact word matches for English terms
        words = sex_lower.split()
        has_male_exact = 'male' in words or any(term in sex_lower for term in male_terms)
        has_female_exact = 'female' in words or any(term in sex_lower for term in female_terms)
        
        if has_male_exact and has_female_exact:
            return "mixed"
        elif has_male_exact:
            return "male"
        elif has_female_exact:
            return "female"
        
        # Mixed indicators
        if any(term in sex_lower for term in ['mixed', '混合', '雌雄', '雌雄各半']):
            return "mixed"
        
        return "未说明"


class DataNormalizer:
    """Main data normalization coordinator."""
    
    def __init__(self):
        self.tumor_volume = TumorVolumeNormalizer()
        self.weight = WeightNormalizer()
        self.organ_mass = OrganMassNormalizer()
        self.dose = DoseNormalizer()
        self.timeline = TimelineNormalizer()
        self.strain = StrainNormalizer()
        self.sex = SexNormalizer()
    
    def normalize_all_fields(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply normalization to all applicable fields in a data dictionary."""
        normalized = data_dict.copy()
        
        # Normalize specific fields if present
        if 'tumor_volume' in normalized:
            normalized['tumor_volume_normalized'] = self.tumor_volume.normalize_volume(
                normalized['tumor_volume']
            )
        
        if 'body_weight' in normalized:
            normalized['body_weight_normalized'] = self.weight.normalize_weight(
                normalized['body_weight']
            )
        
        if 'organ_mass' in normalized:
            normalized['organ_mass_normalized'] = self.organ_mass.normalize_organ_mass(
                normalized['organ_mass']
            )
        
        if 'dose' in normalized:
            dose_mg_kg, freq_norm, daily_equiv = self.dose.normalize_dose(normalized['dose'])
            normalized['dose_mg_per_kg'] = dose_mg_kg
            normalized['dose_frequency_norm'] = freq_norm
            normalized['daily_equiv_mg_per_kg'] = daily_equiv
        
        if 'timepoint' in normalized:
            normalized['canonical_day'] = self.timeline.normalize_timepoint(
                normalized['timepoint']
            )
        
        if 'strain' in normalized:
            raw_strain, norm_strain = self.strain.normalize_strain(normalized['strain'])
            normalized['strain_raw'] = raw_strain
            normalized['strain_norm'] = norm_strain
        
        if 'sex' in normalized:
            normalized['sex_norm'] = self.sex.normalize_sex(normalized['sex'])
        
        # Add metadata
        normalized['units_version'] = "v1.0"
        
        return normalized
