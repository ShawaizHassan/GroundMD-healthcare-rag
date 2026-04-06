import re

class PrivacyFilter:
    def __init__(self):
        self.patient_mapping = {}
        self.counter = 1
        self.patterns = {
            'mrn': r'MRN\s*[:]?\s*(\d+)',
            'name': r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',
        }
    
    def detect_phi(self, text: str):
        detected = []
        mrn_match = re.search(self.patterns['mrn'], text, re.IGNORECASE)
        if mrn_match:
            detected.append(('MRN', mrn_match.group(1)))
        name_match = re.search(self.patterns['name'], text)
        if name_match:
            detected.append(('NAME', name_match.group(1)))
        return detected
    
    def anonymize(self, text: str):
        anonymized = text
        mrn_match = re.search(self.patterns['mrn'], anonymized, re.IGNORECASE)
        if mrn_match:
            mrn = mrn_match.group(1)
            placeholder = f'PATIENT_{self.counter}'
            self.patient_mapping[placeholder] = {'mrn': mrn}
            anonymized = anonymized.replace(f"MRN {mrn}", placeholder)
            anonymized = anonymized.replace(f"MRN:{mrn}", placeholder)
            self.counter += 1
        name_match = re.search(self.patterns['name'], anonymized)
        if name_match:
            name = name_match.group(1)
            placeholder = f'PATIENT_{self.counter}'
            self.patient_mapping[placeholder] = {'name': name}
            anonymized = anonymized.replace(name, placeholder)
            self.counter += 1
        return anonymized
    
    def deanonymize(self, text: str):
        result = text
        for placeholder, info in self.patient_mapping.items():
            if 'name' in info:
                result = result.replace(placeholder, info['name'])
            if 'mrn' in info:
                result = result.replace(placeholder, f"MRN {info['mrn']}")
        return result
    
    def clear_mapping(self):
        self.patient_mapping = {}
        self.counter = 1