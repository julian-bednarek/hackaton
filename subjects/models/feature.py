from django.db import models
from .subject import Subject
from .affect_condition import AffectCondition

class PhysiologicalFeature(models.Model):
    
    DEVICE_CHOICES = [
        ('CHEST', 'RespiBAN (Chest)'),
        ('WRIST', 'Empatica E4 (Wrist)'),
    ]

    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    condition = models.ForeignKey(
        AffectCondition, 
        on_delete=models.PROTECT, 
        limit_choices_to={'condition_id__in': [1, 2, 3]},
    )
    
    device_location = models.CharField(
        max_length=5,
        choices=DEVICE_CHOICES,
    )

    time_window_start_sec = models.DecimalField(
        max_digits=10, 
        decimal_places=3, 
    )
    window_length_sec = models.DecimalField(
        max_digits=5, 
        decimal_places=2, 
        default=30.0,
    )

    hrv_rmssd = models.DecimalField(max_digits=10, decimal_places=5, null=True, blank=True)
    mean_hr = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    eda_mean = models.DecimalField(max_digits=10, decimal_places=5, null=True, blank=True)
    
    acc_std_x = models.DecimalField(max_digits=10, decimal_places=5, null=True, blank=True) 
    bvp_mean = models.DecimalField(max_digits=10, decimal_places=5, null=True, blank=True) 

    model_predicted_stress_prob = models.DecimalField(
        max_digits=4, 
        decimal_places=3, 
        null=True, blank=True,
    )

    class Meta:
        unique_together = ('subject', 'time_window_start_sec', 'device_location')

    def __str__(self):
        return f"{self.subject.code} Feature Window at {self.time_window_start_sec}s"