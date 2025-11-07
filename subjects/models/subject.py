from django.db import models
from django.core.validators import RegexValidator

id_validator = RegexValidator(
    regex=r'^S\d{4}$',
    message='ID must be in the format of S followed by four digits.'
)

class Subject(models.Model):

    class Meta:
        verbose_name = "Subject"
        verbose_name_plural = "Subjects"

    code = models.CharField(
        max_length=10,
        validators=[id_validator], 
        unique=True
    )
    age = models.PositiveIntegerField()
    male = models.BooleanField()
    right_handed = models.BooleanField()

    drank_coffee_today = models.BooleanField()
    drank_coffee_last_hour = models.BooleanField()
    did_sports_today = models.BooleanField()
    felt_ill_today = models.BooleanField()

    additional_notes = models.TextField(blank=True)