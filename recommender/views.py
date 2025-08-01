from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from . import Model  # Make sure this doesn't execute on import

@csrf_exempt
def generate_recommendation_from_input(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            interest = data.get("interest")
            qualification = data.get("qualification")

            if not interest or not qualification:
                return JsonResponse({"error": "Both interest and qualification are required."}, status=400)

            result = Model.generate_recommendation_from_input(interest, qualification)
            return JsonResponse(result, safe=False, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"message": "Send a POST request."}, status=405)
