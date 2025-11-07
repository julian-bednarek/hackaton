import os
from subjects.models import Subject

class ReadmeImporter:
    FIELD_MAPPINGS = {
        'age': 'age', 
        'gender': 'male', 
        'dominant hand': 'right_handed', 
        'did you drink coffee today?': 'drank_coffee_today',
        'did you drink coffee within the last hour?': 'drank_coffee_last_hour',
        'did you do any sports today?': 'did_sports_today',
        'do you feel ill today?': 'felt_ill_today',
    }

    def __init__(self, subject_code, data_directory='.'):
        self.subject_code = subject_code
        self.data_directory = data_directory
        self.filename = f'{subject_code}_readme.txt'
        self.filepath = os.path.join(data_directory, subject_code, self.filename)

    def _read_file_content(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Readme file not found at {self.filepath}")
        with open(self.filepath, 'r') as f:
            return f.read()

    def _parse_content(self, content):
        data = {}
        lines = content.split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip().lower()
                
                if key in self.FIELD_MAPPINGS:
                    model_field = self.FIELD_MAPPINGS[key]
                    
                    if model_field == 'age':
                        try:
                            data[model_field] = int(value)
                        except ValueError:
                            pass
                    elif model_field == 'male':
                        data[model_field] = (value == 'male')
                    elif model_field == 'right_handed':
                        data[model_field] = (value == 'right')
                    else:
                        data[model_field] = (value == 'yes')

        notes = ''
        try:
            notes_section = content.split('### Additional notes ###')[-1].strip()
            notes = notes_section.lstrip('-').strip()
        except:
            pass
            
        data['additional_notes'] = notes
        return data

    def import_and_update(self):
        try:
            content = self._read_file_content()
        except FileNotFoundError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Error reading file content: {e}"

        parsed_data = self._parse_content(content)
        
        try:
            subject, created = Subject.objects.get_or_create(
                code=self.subject_code,
                defaults={'age': parsed_data.get('age', 0), 'male': False, 'right_handed': False}
            )
            
            for field, value in parsed_data.items():
                setattr(subject, field, value)
                
            subject.save()

            action = "created" if created else "updated"
            return True, f"Subject {self.subject_code} successfully {action} with readme data."

        except Exception as e:
            return False, f"Database update error for {self.subject_code}: {e}"