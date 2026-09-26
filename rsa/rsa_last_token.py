"""
RSA at the last prompt token (Sec. 3.3, Fig. 2c / 2d).

Hidden states are read at the final token of the prompt (the position from
which the answer is generated) from every layer's attention output.

Example
    python rsa/rsa_last_token.py --json_path json_files/rsa/aavr.json \
        --save_path exp/rsa/aavr/last
"""
import rsa_common as C


if __name__ == "__main__":
    args = C.build_arg_parser(__doc__).parse_args()
    C.run(args, C.last_token_ids, tag="last token")
