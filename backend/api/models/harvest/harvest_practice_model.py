from django.db import models

from api.models.variety_model import Variety


class HarvestPractice(models.Model):
    """ Holds harvest practices """

    practice_name = models.CharField(max_length=100, null=False, blank=False)
    description = models.TextField()

    variety = models.ForeignKey(Variety, default=None, null=True, blank=True, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.variety.variety} - {self.practice_name}'



"""
Here are some common banana post-harvest practices:

Harvesting: Bananas are usually harvested when they are mature but still green. The fruits are carefully cut from the plant using a sharp knife or pruning shears.

Handling: Bananas should be handled with care to avoid bruising or damage. They are usually packed in containers or crates that provide proper ventilation and support.

Sorting and Grading: Bananas are sorted and graded based on their size, color, and quality. This helps in categorizing them for different market segments.

Washing: Bananas are often washed to remove dirt, dust, and any surface contaminants. However, excessive washing should be avoided as it can affect the fruit's quality and increase the risk of fungal growth.

Drying: After washing, bananas are dried using clean, absorbent materials or by air-drying. Excess moisture can promote spoilage, so it's important to ensure they are adequately dried.

Packaging: Bananas are usually packed in bunches or individually wrapped in packaging materials to protect them during transportation and storage. Packaging materials should provide proper ventilation to prevent moisture buildup.

Temperature and Humidity Control: Bananas are sensitive to temperature and humidity. They are typically stored and transported at specific temperature ranges to slow down ripening and prolong their shelf life.

Ripening: Depending on the market requirements, bananas may be ripened artificially using ethylene gas or naturally by storing them at room temperature. Ripening chambers or rooms are used to control the process.

Quality Control: Regular inspections are performed to ensure the quality and condition of the bananas. Any damaged or spoiled fruits are removed to prevent further deterioration.

Transportation: Bananas are transported in specialized vehicles or containers equipped with proper ventilation and temperature control. This helps maintain the desired conditions during transit.
"""