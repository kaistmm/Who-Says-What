"""
RSA at the anchor-attribute token (Sec. 3.3, Fig. 2a / 2b).

The anchor attribute is the word given in the text prompt: the spoken country
name for AAVR (json key ``target_animal`` holds the country) or the animal
name for VAAR.  Hidden states are read at that token (averaged over its
sub-word tokens) from every layer's attention output.

Example
    python rsa/rsa_anchor_token.py --json_path json_files/rsa/aavr.json \
        --save_path exp/rsa/aavr/anchor
"""
import rsa_common as C


if __name__ == "__main__":
    args = C.build_arg_parser(__doc__).parse_args()
    C.run(args, C.find_anchor_token_ids, tag="anchor token")
