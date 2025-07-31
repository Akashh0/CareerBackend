from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from . import Model  # Your updated Model.py using dotenv and Gemini

@csrf_exempt
def generate_recommendation_from_input(request):
    if request.method == 'POST':
        try:
            # Parse request body
            data = json.loads(request.body)
            interest = data.get("interest")
            qualification = data.get("qualification")

            # Validate inputs
            if not interest or not qualification:
                return JsonResponse({"error": "Both interest and qualification are required."}, status=400)

            # Run the model's function to get results
            result = Model.generate_recommendation_from_input(interest, qualification)

            # Return the structured response
            return JsonResponse(result, safe=False, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"message": "Send a POST request."}, status=405)
