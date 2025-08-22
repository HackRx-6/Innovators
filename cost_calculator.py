import json

def calculate_llm_costs(log_data: list) -> dict:
    """
    Calculates the token usage and cost for each operation in a list of LLM log entries,
    and provides a total summary.

    Args:
        log_data (list): A list of JSON objects, where each object represents a single
                         LLM operation and its metadata.

    Returns:
        dict: A dictionary containing a list of per-operation costs and a summary
              of total tokens and cost.
    """
    
    # Define model pricing per 1000 tokens. This model is fictional, so rates are illustrative.
    # Prices are in USD.
    PRICING = {
        "gpt-4.1-mini-2025-04-14": {
            "prompt_rate": 0.0005,  # $0.0005 per 1k prompt tokens
            "completion_rate": 0.0015, # $0.0015 per 1k completion tokens
        }
    }

    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_all_tokens = 0
    operation_costs = []

    for entry in log_data:
        # Get the operation name, which is the first key that isn't 'timestamp'.
        operation_name = next(key for key in entry.keys() if key != "timestamp")
        operation_data = entry[operation_name]

        try:
            # Extract token counts from the 'response_metadata' field
            token_usage = operation_data["response_metadata"]["token_usage"]
            prompt_tokens = token_usage["prompt_tokens"]
            completion_tokens = token_usage["completion_tokens"]
            total_tokens = token_usage["total_tokens"]
            model_name = operation_data["response_metadata"]["model_name"]
            
            # Retrieve rates for the specific model
            rates = PRICING.get(model_name, {"prompt_rate": 0, "completion_rate": 0})
            prompt_rate = rates["prompt_rate"]
            completion_rate = rates["completion_rate"]

            # Calculate the cost for the current operation
            cost = (prompt_tokens / 1000) * prompt_rate + (completion_tokens / 1000) * completion_rate

            # Add to running totals
            total_cost += cost
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_all_tokens += total_tokens

            # Store the results for this specific operation
            operation_costs.append({
                "operation": operation_name,
                "model_name": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(cost, 6)
            })

        except (KeyError, TypeError) as e:
            print(f"Skipping entry for '{operation_name}' due to missing data: {e}")
            continue

    # Prepare the final output dictionary
    output_data = {
        "operations": operation_costs,
        "summary": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_all_tokens,
            "total_cost_usd": round(total_cost, 6)
        }
    }

    return output_data

# Example Usage:
# The provided JSON data is a list of dictionaries.
input_json_data = [
  {
    "analyse_document_content_rag_orchestrator": {
      "content": "set_goal_agent",
      "additional_kwargs": {
        "refusal": None
      },
      "response_metadata": {
        "token_usage": {
          "completion_tokens": 4,
          "prompt_tokens": 189,
          "total_tokens": 193,
          "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0
          },
          "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 0
          }
        },
        "model_name": "gpt-4.1-mini-2025-04-14",
        "system_fingerprint": "fp_4f3d32ad4e",
        "id": "chatcmpl-C7CUYTqdrog0ouCKOpqbo9VLby1DC",
        "service_tier": None,
        "prompt_filter_results": [
          {
            "prompt_index": 0,
            "content_filter_results": {
              "hate": {
                "filtered": False,
                "severity": "safe"
              },
              "jailbreak": {
                "filtered": False,
                "detected": False
              },
              "self_harm": {
                "filtered": False,
                "severity": "safe"
              },
              "sexual": {
                "filtered": False,
                "severity": "safe"
              },
              "violence": {
                "filtered": False,
                "severity": "safe"
              }
            }
          }
        ],
        "finish_reason": "stop",
        "logprobs": None,
        "content_filter_results": {
          "hate": {
            "filtered": False,
            "severity": "safe"
          },
          "protected_material_code": {
            "filtered": False,
            "detected": False
          },
          "protected_material_text": {
            "filtered": False,
            "detected": False
          },
          "self_harm": {
            "filtered": False,
            "severity": "safe"
          },
          "sexual": {
            "filtered": False,
            "severity": "safe"
          },
          "violence": {
            "filtered": False,
            "severity": "safe"
          }
        }
      },
      "type": "ai",
      "name": None,
      "id": "run--c2e43902-e6d7-460b-9bbc-76dec267b9e5-0",
      "example": False,
      "tool_calls": [],
      "invalid_tool_calls": [],
      "usage_metadata": {
        "input_tokens": 189,
        "output_tokens": 4,
        "total_tokens": 193,
        "input_token_details": {
          "audio": 0,
          "cache_read": 0
        },
        "output_token_details": {
          "audio": 0,
          "reasoning": 0
        }
      }
    },
    "timestamp": "2025-08-22T08:42:29.458882+05:30"
  },
  {
    "set_goal_agent_rag_orchestrator": {
      "content": "{\n \"goal\": \"Find the flight number.\",\n \"steps\": [\n  {\n   \"step\": 1,\n   \"description\": \"Call the API at https://register.hackrx.in/submissions/myFavouriteCity to get the city name.\"\n  },\n  {\n   \"step\": 2,\n   \"description\": \"Use the city name from step 1 to find the corresponding landmark from Sachin's travel notes in the document.\"\n  },\n  {\n   \"step\": 3,\n   \"description\": \"Based on the landmark found in step 2, call the appropriate flight number API as per the instructions: if Gateway of India, call getFirstCityFlightNumber; if Taj Mahal, call getSecondCityFlightNumber; if Eiffel Tower, call getThirdCityFlightNumber; if Big Ben, call getFourthCityFlightNumber; otherwise, call getFifthCityFlightNumber.\"\n  },\n  {\n   \"step\": 4,\n   \"description\": \"Return the flight number received from the API call in step 3 as the final answer.\"\n  }\n ]\n}",
      "additional_kwargs": {
        "refusal": None
      },
      "response_metadata": {
        "token_usage": {
          "completion_tokens": 223,
          "prompt_tokens": 1402,
          "total_tokens": 1625,
          "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0
          },
          "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 0
          }
        },
        "model_name": "gpt-4.1-mini-2025-04-14",
        "system_fingerprint": "fp_4f3d32ad4e",
        "id": "chatcmpl-C7CUZFg6OHrEKNi4gw2SGJ7F0FVgT",
        "service_tier": None,
        "prompt_filter_results": [
          {
            "prompt_index": 0,
            "content_filter_results": {
              "hate": {
                "filtered": False,
                "severity": "safe"
              },
              "jailbreak": {
                "filtered": False,
                "detected": False
              },
              "self_harm": {
                "filtered": False,
                "severity": "safe"
              },
              "sexual": {
                "filtered": False,
                "severity": "safe"
              },
              "violence": {
                "filtered": False,
                "severity": "safe"
              }
            }
          }
        ],
        "finish_reason": "stop",
        "logprobs": None,
        "content_filter_results": {
          "hate": {
            "filtered": False,
            "severity": "safe"
          },
          "protected_material_code": {
            "filtered": False,
            "detected": False
          },
          "protected_material_text": {
            "filtered": False,
            "detected": False
          },
          "self_harm": {
            "filtered": False,
            "severity": "safe"
          },
          "sexual": {
            "filtered": False,
            "severity": "safe"
          },
          "violence": {
            "filtered": False,
            "severity": "safe"
          }
        }
      },
      "type": "ai",
      "name": None,
      "id": "run--6bf94fa4-5c96-4374-bd83-866c79478df1-0",
      "example": False,
      "tool_calls": [],
      "invalid_tool_calls": [],
      "usage_metadata": {
        "input_tokens": 1402,
        "output_tokens": 223,
        "total_tokens": 1625,
        "input_token_details": {
          "audio": 0,
          "cache_read": 0
        },
        "output_token_details": {
          "audio": 0,
          "reasoning": 0
        }
      }
    },
    "timestamp": "2025-08-22T08:42:32.724115+05:30"
  },
  {
    "perform_action_rag_orchestrator": {
      "content": "",
      "additional_kwargs": {
        "tool_calls": [
          {
            "id": "call_zdbzqHGxAy610yxcWRFovtiq",
            "function": {
              "arguments": "{\"url\":\"https://register.hackrx.in/submissions/myFavouriteCity\"}",
              "name": "get_from_endpoint"
            },
            "type": "function"
          }
        ],
        "refusal": None
      },
      "response_metadata": {
        "token_usage": {
          "completion_tokens": 27,
          "prompt_tokens": 1409,
          "total_tokens": 1436,
          "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0
          },
          "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 0
          }
        },
        "model_name": "gpt-4.1-mini-2025-04-14",
        "system_fingerprint": "fp_4f3d32ad4e",
        "id": "chatcmpl-C7CUc96Pxg1DioZMwaENnb4GS8nm7",
        "service_tier": None,
        "prompt_filter_results": [
          {
            "prompt_index": 0,
            "content_filter_results": {}
          }
        ],
        "finish_reason": "tool_calls",
        "logprobs": None,
        "content_filter_results": {}
      },
      "type": "ai",
      "name": None,
      "id": "run--d3be944b-bd85-474e-976d-7115d4781589-0",
      "example": False,
      "tool_calls": [
        {
          "name": "get_from_endpoint",
          "args": {
            "url": "https://register.hackrx.in/submissions/myFavouriteCity"
          },
          "id": "call_zdbzqHGxAy610yxcWRFovtiq",
          "type": "tool_call"
        }
      ],
      "invalid_tool_calls": [],
      "usage_metadata": {
        "input_tokens": 1409,
        "output_tokens": 27,
        "total_tokens": 1436,
        "input_token_details": {
          "audio": 0,
          "cache_read": 0
        },
        "output_token_details": {
          "audio": 0,
          "reasoning": 0
        }
      }
    },
    "timestamp": "2025-08-22T08:42:33.956297+05:30"
  }
]

# Run the function with the provided data
cost_breakdown_result = calculate_llm_costs(input_json_data)

# Print the resulting JSON
print(json.dumps(cost_breakdown_result, indent=2))