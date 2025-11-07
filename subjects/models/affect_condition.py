from django.db import models

class AffectCondition(models.Model):
    CONDITION_CHOICES = [
        (0, 'Not Defined / Transient'),
        (1, 'Baseline'),
        (2, 'Stress'),
        (3, 'Amusement'),
        (4, 'Meditation'),
        (5, 'Ignore_5'),
        (6, 'Ignore_6'),
        (7, 'Ignore_7'),
    ]
    
    condition_id = models.PositiveSmallIntegerField(
        choices=CONDITION_CHOICES,
        unique=True,
        primary_key=True,
    )
    name = models.CharField(max_length=50)

    class Meta:
        ordering = ['condition_id']

    def __str__(self):
        return self.name