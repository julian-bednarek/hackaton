# models.py (Consolidated Self-Report Model without comments)

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

from .subject import Subject
# Assuming Subject and AffectCondition models exist and are imported.

# --- Scale Definitions ---
PANAS_SSSQ_SCALE = [(i, str(i)) for i in range(1, 6)] 
STAI_SCALE = [(i, str(i)) for i in range(1, 5)] 
SAM_SCALE = [(i, str(i)) for i in range(1, 10)] 

class SelfReportScore(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    # condition = models.ForeignKey(
    #     AffectCondition, 
    #     on_delete=models.PROTECT,
    # )
    
    class Meta:
        unique_together = ('subject', 'condition')
        
    active = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    distressed = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    interested = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    inspired = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    annoyed = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    strong = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    guilty = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    scared = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    hostile = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    excited = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    proud = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    irritable = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    enthusiastic = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    ashamed = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    alert = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    nervous = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    determined = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    attentive = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    jittery = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    afraid = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    
    stressed = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    frustrated = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    happy = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)
    sad = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE)

    feel_at_ease = models.PositiveSmallIntegerField(choices=STAI_SCALE)
    feel_nervous_stai = models.PositiveSmallIntegerField(choices=STAI_SCALE)
    am_jittery_stai = models.PositiveSmallIntegerField(choices=STAI_SCALE)
    am_relaxed = models.PositiveSmallIntegerField(choices=STAI_SCALE)
    am_worried = models.PositiveSmallIntegerField(choices=STAI_SCALE)
    feel_pleasant = models.PositiveSmallIntegerField(choices=STAI_SCALE)

    sam_valence = models.PositiveSmallIntegerField(choices=SAM_SCALE)
    sam_arousal = models.PositiveSmallIntegerField(choices=SAM_SCALE)

    angry_sssq = models.PositiveSmallIntegerField(
        choices=PANAS_SSSQ_SCALE, 
        null=True, blank=True
    )
    irritable_sssq = models.PositiveSmallIntegerField(
        choices=PANAS_SSSQ_SCALE, 
        null=True, blank=True
    )
    committed_to_goals = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE, null=True, blank=True)
    wanted_to_succeed = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE, null=True, blank=True)
    motivated_to_do_task = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE, null=True, blank=True)
    reflected_about_self = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE, null=True, blank=True)
    worried_about_others_opinion = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE, null=True, blank=True)
    concerned_about_impression = models.PositiveSmallIntegerField(choices=PANAS_SSSQ_SCALE, null=True, blank=True)

    # def __str__(self):
    #     return f"{self.subject.code} Report after {self.condition.name}"