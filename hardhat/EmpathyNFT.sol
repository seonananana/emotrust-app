// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * 목적: 검열/PII 마스킹/점수 통과(>=0.70)한 글을 tokenURI로 기록.
 * - owner(=배포자)만 민팅 가능: 서버가 PRIVATE_KEY로 서명해서 실행.
 * - tokenURI(string)에는 data:application/json;base64, ... 메타데이터를 넣음.
 */
contract EmpathyNFT is ERC721URIStorage, Ownable {
    uint256 private _id;

    constructor() ERC721("EmpathyFinance", "EMPF") Ownable(msg.sender) {}

    function safeMint(address to, string memory uri) public onlyOwner returns (uint256) {
        _id += 1;
        uint256 tokenId = _id;
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, uri);
        return tokenId;
    }
}
